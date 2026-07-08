
from __future__ import annotations
import sys
import time
import copy
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from irp_heuristic import IRPHeuristic, DEFAULT_COVERAGE_HOURS




SAFETY_STOCK_MULTIPLIERS     = [0.5, 0.75, 1.0, 1.25, 1.5]
TRAILER_CAPACITY_MULTIPLIERS = [0.7, 0.85, 1.0, 1.15, 1.3]
DRIVER_AVAIL_MULTIPLIERS     = [0.7, 0.85, 1.0, 1.15, 1.3]

OUTPUT_DIR = Path('sensitivity_output')

SNAPSHOT_MULTIPLIERS = {
    'safety_stock':        [SAFETY_STOCK_MULTIPLIERS[0],  1.0, SAFETY_STOCK_MULTIPLIERS[-1]],
    'trailer_capacity':    [TRAILER_CAPACITY_MULTIPLIERS[0], 1.0, TRAILER_CAPACITY_MULTIPLIERS[-1]],
    'driver_availability': [DRIVER_AVAIL_MULTIPLIERS[0],  1.0, DRIVER_AVAIL_MULTIPLIERS[-1]],
}
SNAPSHOT_INSTANCES = [
    'Instance_V_1.1',  'Instance_V_1.2',  'Instance_V_1.3',
    'Instance_V_1.4',  'Instance_V_1.5',  'Instance_V_1.6',
    'Instance_V_1.7',  'Instance_V_1.8',  'Instance_V_1.9',
    'Instance_V_1.10', 'Instance_V_1.11',
]




class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(log_path, 'w', encoding='utf-8')

    def __call__(self, msg: str = '') -> None:
        print(msg, flush=True)
        self._fh.write(msg + '\n')
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()




def find_instance_folders(root: Path) -> List[Path]:

    folders = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith('Instance_V_')],
        key=_instance_sort_key_path,
    )
    return folders


def _reset_solve_state(solver: IRPHeuristic) -> None:

    max_loc = max(solver.location_indices)
    solver.delivered     = np.zeros((max_loc + 1, solver.horizon_hours), dtype=float)
    solver.routes        = []
    solver.next_available = {d: 0.0 for d in solver.next_available}


def run_one_solve(solver: IRPHeuristic, label: str, log: Logger) -> dict:

    t0 = time.time()
    try:
        metrics = solver.solve()
    except Exception as ex:
        log(f"      ERROR ({label}): {ex}")
        return {
            'feasible': False, 'LR': None, 'total_cost': None,
            'distance_cost': None, 'time_cost': None, 'total_quantity': None,
            'n_routes': None, 'stockout_hours': None,
            'safety_violation_hours': None, 'overflow_hours': None,
            'wall_time_sec': round(time.time() - t0, 1),
            'error': str(ex),
            '_solver': None,
        }
    elapsed = time.time() - t0
    feasible = (metrics['stockout_hours'] == 0 and
                metrics['tank_overflow_hours'] == 0)
    return {
        'feasible':               feasible,
        'LR':                     metrics['LR'],
        'total_cost':             metrics['total_cost'],
        'distance_cost':          metrics['distance_cost'],
        'time_cost':              metrics['time_cost'],
        'total_quantity':         metrics['total_quantity_delivered'],
        'n_routes':               metrics['num_routes'],
        'stockout_hours':         metrics['stockout_hours'],
        'safety_violation_hours': metrics['safety_violation_hours'],
        'overflow_hours':         metrics['tank_overflow_hours'],
        'wall_time_sec':          round(elapsed, 1),
        'error':                  None,
        '_solver':                solver,   # kept for inventory snapshots
    }




def apply_safety_stock_multiplier(baseline: IRPHeuristic, m: float) -> IRPHeuristic:
    s = copy.deepcopy(baseline)
    _reset_solve_state(s)
    for c in s.customers:
        s.safety[c] = baseline.safety[c] * m
    return s


def apply_trailer_capacity_multiplier(baseline: IRPHeuristic, m: float) -> IRPHeuristic:
    s = copy.deepcopy(baseline)
    _reset_solve_state(s)
    for k in list(s.trailer_capacity.keys()):
        s.trailer_capacity[k] = baseline.trailer_capacity[k] * m
    return s


def apply_driver_availability_multiplier(baseline: IRPHeuristic, m: float) -> IRPHeuristic:

    s = copy.deepcopy(baseline)
    _reset_solve_state(s)

    for drv in list(s.driver_max_drive.keys()):
        s.driver_max_drive[drv] = max(60.0, baseline.driver_max_drive[drv] * m)

    dw = s.driver_windows.copy()
    starts  = dw['start'].values.astype(float)
    ends    = dw['end'].values.astype(float)
    lengths = ends - starts
    dw['end'] = starts + np.maximum(60.0, lengths * m)
    s.driver_windows = dw
    return s




def extract_inventory_snapshot(solver: IRPHeuristic,
                                multiplier: float,
                                parameter: str) -> List[dict]:

    rows      = []
    inv_df    = solver.simulate_inventory()
    hour_cols = [c for c in inv_df.columns if c.startswith('t')]
    horizon   = len(hour_cols)
    step      = max(1, horizon // 80)
    sample_customers = solver.customers[:8]

    for _, row in inv_df[inv_df['location_index'].isin(sample_customers)].iterrows():
        c   = int(row['location_index'])
        cap = solver.capacity[c]
        ss  = solver.safety[c]
        for h in range(0, horizon + 1, step):
            col = f't{h}' if h < horizon else f't{horizon - 1}'
            rows.append({
                'parameter':    parameter,
                'multiplier':   multiplier,
                'instance':     Path(solver.data_dir).name,
                'customer':     c,
                'hour':         h,
                'inventory':    float(row[col]) if col in row.index else np.nan,
                'capacity':     float(cap),
                'safety_stock': float(ss),
            })
    return rows




def run_sweep(baseline: IRPHeuristic,
              mutator,
              multipliers: List[float],
              parameter_name: str,
              snapshot_levels: List[float],
              snapshot_instances: List[str],
              log: Logger) -> tuple:
    sweep_rows, inventory_rows = [], []
    inst_name = Path(baseline.data_dir).name

    for m in multipliers:
        solver_m = mutator(baseline, m)
        result   = run_one_solve(solver_m, f"{parameter_name} x{m:.2f}", log)
        row = {'parameter': parameter_name, 'multiplier': m, 'instance': inst_name}
        for k, v in result.items():
            if not k.startswith('_'):
                row[k] = v
        sweep_rows.append(row)

        if result.get('LR') is not None:
            log(f"      {parameter_name} x{m:.2f}: "
                f"LR={result['LR']:.5f}  routes={result['n_routes']}  "
                f"stockout={result['stockout_hours']}  "
                f"safety_viol={result['safety_violation_hours']}  "
                f"overflow={result['overflow_hours']}  "
                f"feasible={result['feasible']}  "
                f"({result['wall_time_sec']}s)")

        if (m in snapshot_levels
                and inst_name in snapshot_instances
                and result['_solver'] is not None):
            try:
                inv_rows = extract_inventory_snapshot(
                    result['_solver'], m, parameter_name
                )
                inventory_rows.extend(inv_rows)
            except Exception as ex:
                log(f"      WARN: could not extract snapshot at m={m}: {ex}")

    return sweep_rows, inventory_rows




def _instance_sort_key(name: str) -> tuple:
    try:
        suffix = name.replace('Instance_V_', '')
        major, minor = suffix.split('.')
        return (int(major), int(minor))
    except Exception:
        return (99, 99)


def _instance_sort_key_path(p: Path) -> tuple:
    return _instance_sort_key(p.name)


def plot_metric_vs_multiplier(df: pd.DataFrame, metric: str, ylabel: str,
                               xlabel: str, title: str, outfile: Path,
                               log_scale: bool = False) -> None:
    df = df.dropna(subset=[metric]).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6.5))
    instances = sorted(df['instance'].unique(), key=_instance_sort_key)
    cmap = plt.get_cmap('tab20')
    for i, name in enumerate(instances):
        sub = df[df['instance'] == name].sort_values('multiplier')
        ax.plot(sub['multiplier'], sub[metric], marker='o',
                label=name, linewidth=1.5,
                color=cmap(i / max(1, len(instances) - 1)))
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    if log_scale and (df[metric] > 0).all():
        ax.set_yscale('log')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='baseline')
    ax.legend(fontsize=8, loc='center left',
              bbox_to_anchor=(1.02, 0.5), frameon=True, ncol=1)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_inventory_trajectories(inv_df: pd.DataFrame, instance: str,
                                 parameter: str, outfile: Path,
                                 xlabel_suffix: str) -> None:
    sub = inv_df[(inv_df['instance'] == instance) &
                 (inv_df['parameter'] == parameter)]
    if sub.empty:
        return
    customers   = sorted(sub['customer'].unique())[:6]
    multipliers = sorted(sub['multiplier'].unique())
    if not customers or not multipliers:
        return

    n_mult = len(multipliers)
    fig, axes = plt.subplots(n_mult, 1,
                              figsize=(11, 3.0 * n_mult), sharex=True)
    if n_mult == 1:
        axes = [axes]

    cmap = plt.get_cmap('tab10')
    for row_i, m in enumerate(multipliers):
        ax    = axes[row_i]
        sub_m = sub[sub['multiplier'] == m]
        for j, c in enumerate(customers):
            sub_c = sub_m[sub_m['customer'] == c].sort_values('hour')
            if sub_c.empty:
                continue
            color = cmap(j / max(1, len(customers) - 1))
            ax.plot(sub_c['hour'], sub_c['inventory'],
                    label=f'cust {c}', color=color, linewidth=1.4)
            cap = sub_c['capacity'].iloc[0]
            ss  = sub_c['safety_stock'].iloc[0]
            ax.axhline(y=cap,  linestyle=':', color=color, alpha=0.35, linewidth=0.8)
            if ss > 0:
                ax.axhline(y=ss, linestyle='--', color=color, alpha=0.35, linewidth=0.8)
        ax.set_ylabel('Inventory')
        ax.set_title(f'{parameter} multiplier = {m:.2f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.3, linewidth=0.7)
        if row_i == 0:
            ax.legend(fontsize=7, loc='center left',
                      bbox_to_anchor=(1.02, 0.5), ncol=1, frameon=True,
                      title='customer (dotted=cap, dashed=safety)')
    axes[-1].set_xlabel(
        f'Hour into planning horizon ({xlabel_suffix})', fontsize=10)
    fig.suptitle(
        f'Inventory trajectories — {instance} — sweeping {parameter}',
        fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)




def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log = Logger(OUTPUT_DIR / 'console_log.txt')
    log("=" * 70)
    log("Case 1: Liquid Oxygen IRP — Sensitivity Analysis")
    log("=" * 70)

    root    = Path('.')
    folders = find_instance_folders(root)
    if not folders:
        log("ERROR: No Instance_V_*.* folders found. "
            "Place this file in the same directory as the instance folders.")
        log.close()
        return

    log(f"Discovered {len(folders)} instance folders:")
    for f in folders:
        log(f"  - {f.name}")
    log("")
    log(f"Sweep multipliers:")
    log(f"  Safety stock:        {SAFETY_STOCK_MULTIPLIERS}")
    log(f"  Trailer capacity:    {TRAILER_CAPACITY_MULTIPLIERS}")
    log(f"  Driver availability: {DRIVER_AVAIL_MULTIPLIERS}")
    log(f"Snapshot instances for inventory plots: {SNAPSHOT_INSTANCES}")
    log("")

    overall_t0 = time.time()
    safety_rows, trailer_rows, driver_rows = [], [], []
    inventory_rows: List[dict] = []

    for folder in folders:
        log(f"\n{'=' * 70}")
        log(f"INSTANCE: {folder.name}")
        log(f"{'=' * 70}")

        # Load data ONCE and solve ONCE → this is the exact baseline
        try:
            baseline = IRPHeuristic(
                data_dir=str(folder),
                suffix=None,
                output_dir='.',
                coverage_hours=DEFAULT_COVERAGE_HOURS,
            )
            baseline.solve()   # populates delivered / routes
        except Exception as ex:
            log(f"  ERROR loading/solving instance: {ex}")
            continue

        log(f"  customers={len(baseline.customers)}  "
            f"drivers={len(set(baseline.driver_trailer.keys()))}  "
            f"trailers={len(baseline.trailer_capacity)}  "
            f"horizon={baseline.horizon_hours}h")

        log(f"\n  [1/3] Sweeping safety stock multipliers...")
        rows, inv = run_sweep(
            baseline, apply_safety_stock_multiplier,
            SAFETY_STOCK_MULTIPLIERS, 'safety_stock',
            SNAPSHOT_MULTIPLIERS['safety_stock'], SNAPSHOT_INSTANCES, log
        )
        safety_rows.extend(rows)
        inventory_rows.extend(inv)

        log(f"\n  [2/3] Sweeping trailer capacity multipliers...")
        rows, inv = run_sweep(
            baseline, apply_trailer_capacity_multiplier,
            TRAILER_CAPACITY_MULTIPLIERS, 'trailer_capacity',
            SNAPSHOT_MULTIPLIERS['trailer_capacity'], SNAPSHOT_INSTANCES, log
        )
        trailer_rows.extend(rows)
        inventory_rows.extend(inv)

        log(f"\n  [3/3] Sweeping driver availability multipliers...")
        rows, inv = run_sweep(
            baseline, apply_driver_availability_multiplier,
            DRIVER_AVAIL_MULTIPLIERS, 'driver_availability',
            SNAPSHOT_MULTIPLIERS['driver_availability'], SNAPSHOT_INSTANCES, log
        )
        driver_rows.extend(rows)
        inventory_rows.extend(inv)


    log(f"\n\n{'=' * 70}")
    log("WRITING CSV FILES")
    log(f"{'=' * 70}")

    safety_df    = pd.DataFrame(safety_rows)
    trailer_df   = pd.DataFrame(trailer_rows)
    driver_df    = pd.DataFrame(driver_rows)
    inventory_df = pd.DataFrame(inventory_rows)

    safety_df.to_csv(OUTPUT_DIR / 'safety_stock_sweep.csv',        index=False)
    trailer_df.to_csv(OUTPUT_DIR / 'trailer_capacity_sweep.csv',   index=False)
    driver_df.to_csv(OUTPUT_DIR / 'driver_availability_sweep.csv', index=False)
    if not inventory_df.empty:
        inventory_df.to_csv(OUTPUT_DIR / 'inventory_snapshots.csv', index=False)

    log(f"  CSVs saved to {OUTPUT_DIR.resolve()}/")
    log(f"  - safety_stock_sweep.csv         ({len(safety_df)} rows)")
    log(f"  - trailer_capacity_sweep.csv     ({len(trailer_df)} rows)")
    log(f"  - driver_availability_sweep.csv  ({len(driver_df)} rows)")
    log(f"  - inventory_snapshots.csv        ({len(inventory_df)} rows)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log("GENERATING PLOTS")
    log(f"{'=' * 70}")

    plot_metric_vs_multiplier(
        safety_df, 'LR', 'Logistics Ratio (LR)',
        'Safety stock multiplier', 'LR vs Safety Stock Multiplier',
        OUTPUT_DIR / 'plot_lr_vs_safety_stock.png')
    plot_metric_vs_multiplier(
        trailer_df, 'LR', 'Logistics Ratio (LR)',
        'Trailer capacity multiplier', 'LR vs Trailer Capacity Multiplier',
        OUTPUT_DIR / 'plot_lr_vs_trailer_capacity.png')
    plot_metric_vs_multiplier(
        driver_df, 'LR', 'Logistics Ratio (LR)',
        'Driver availability multiplier', 'LR vs Driver Availability Multiplier',
        OUTPUT_DIR / 'plot_lr_vs_driver_availability.png')

    plot_metric_vs_multiplier(
        safety_df, 'stockout_hours', 'Stockout hours',
        'Safety stock multiplier', 'Stockout hours vs Safety Stock Multiplier',
        OUTPUT_DIR / 'plot_stockouts_vs_safety_stock.png')
    plot_metric_vs_multiplier(
        trailer_df, 'stockout_hours', 'Stockout hours',
        'Trailer capacity multiplier', 'Stockout hours vs Trailer Capacity Multiplier',
        OUTPUT_DIR / 'plot_stockouts_vs_trailer_capacity.png')
    plot_metric_vs_multiplier(
        driver_df, 'stockout_hours', 'Stockout hours',
        'Driver availability multiplier', 'Stockout hours vs Driver Availability Multiplier',
        OUTPUT_DIR / 'plot_stockouts_vs_driver_availability.png')

    plot_metric_vs_multiplier(
        safety_df, 'safety_violation_hours', 'Safety-stock violation hours',
        'Safety stock multiplier', 'Safety-stock violations vs Safety Stock Multiplier',
        OUTPUT_DIR / 'plot_safety_violations_vs_safety_stock.png')

    plot_metric_vs_multiplier(
        trailer_df, 'n_routes', 'Number of routes used',
        'Trailer capacity multiplier', 'Routes vs Trailer Capacity Multiplier',
        OUTPUT_DIR / 'plot_routes_vs_trailer_capacity.png')
    plot_metric_vs_multiplier(
        driver_df, 'n_routes', 'Number of routes used',
        'Driver availability multiplier', 'Routes vs Driver Availability Multiplier',
        OUTPUT_DIR / 'plot_routes_vs_driver_availability.png')

    if not inventory_df.empty:
        for inst_name in SNAPSHOT_INSTANCES:
            for param in ['safety_stock', 'trailer_capacity', 'driver_availability']:
                outpath = OUTPUT_DIR / f'plot_inventory_{inst_name}_{param}.png'
                plot_inventory_trajectories(
                    inventory_df, inst_name, param, outpath,
                    xlabel_suffix=inst_name)

    plot_files = sorted(OUTPUT_DIR.glob('plot_*.png'))
    log(f"  {len(plot_files)} plots saved:")
    for p in plot_files:
        log(f"    - {p.name}")


    log(f"\n{'=' * 70}")
    log("SUMMARY: LR change vs baseline (multiplier = 1.0)")
    log(f"{'=' * 70}")
    summary_rows = []
    for param_name, df in [('safety_stock',        safety_df),
                            ('trailer_capacity',    trailer_df),
                            ('driver_availability', driver_df)]:
        log(f"\n{param_name}:")
        log(f"  {'Instance':<20} {'min mult':>9} {'min LR':>10}  "
            f"{'baseline':>10}  {'max mult':>9} {'max LR':>10}  "
            f"{'%-change':>10}")
        for inst_name in sorted(df['instance'].unique(), key=_instance_sort_key):
            sub = df[df['instance'] == inst_name].dropna(subset=['LR'])
            if sub.empty:
                continue
            mult_min  = sub['multiplier'].min()
            mult_max  = sub['multiplier'].max()
            lr_min    = sub.loc[sub['multiplier'] == mult_min, 'LR'].iloc[0]
            lr_max    = sub.loc[sub['multiplier'] == mult_max, 'LR'].iloc[0]
            base_rows = sub.loc[sub['multiplier'] == 1.0, 'LR']
            lr_base   = base_rows.iloc[0] if len(base_rows) > 0 else float('nan')
            pct_range = ((lr_max - lr_min) / lr_base * 100.0
                         if (lr_base and not np.isnan(lr_base)) else float('nan'))
            log(f"  {inst_name:<20} {mult_min:>9.2f} {lr_min:>10.5f}  "
                f"{lr_base:>10.5f}  {mult_max:>9.2f} {lr_max:>10.5f}  "
                f"{pct_range:>9.2f}%")
            summary_rows.append({
                'parameter':     param_name,
                'instance':      inst_name,
                'min_multiplier': mult_min,
                'LR_at_min':     lr_min,
                'baseline_LR':   lr_base,
                'max_multiplier': mult_max,
                'LR_at_max':     lr_max,
                'LR_pct_range':  pct_range,
            })

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / 'summary_table.csv', index=False)
    log(f"\n  Summary table written to summary_table.csv")

    total = time.time() - overall_t0
    log(f"\n{'=' * 70}")
    log(f"Total wall-clock time: {total:.0f} sec ({total/60:.1f} min)")
    log(f"{'=' * 70}")
    log("\nDone.")
    log.close()


if __name__ == '__main__':
    main()
