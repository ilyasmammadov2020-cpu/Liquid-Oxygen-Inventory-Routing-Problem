
from __future__ import annotations

import os
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd


# ─────────────────────────────────────────
#  Tunable parameters
# ─────────────────────────────────────────
DEFAULT_COVERAGE_HOURS  = 96
MIN_DELIVERY_FRAC       = 0.04
MIN_DELIVERY_ABS        = 50.0
URGENCY_WEIGHT          = 1_000_000.0
MAX_CUSTOMERS_PER_ROUTE = 20
TWO_OPT_SWEEPS          = 3
EPS                     = 1e-9


@dataclass
class RouteMetrics:
    sequence: List[int]
    arrivals_min: Dict[int, float]
    arrivals_hour: Dict[int, int]
    distance: float
    drive_time: float
    duration: float
    end_time: float


@dataclass
class RouteResult:
    route_id: int
    driver: int
    trailer: int
    window_start: float
    window_end: float
    actual_start: float
    sequence: List[int]
    deliveries: Dict[int, float]
    arrivals_min: Dict[int, float]
    arrivals_hour: Dict[int, int]
    distance: float
    drive_time: float
    duration: float
    end_time: float
    distance_cost: float
    time_cost: float

    @property
    def total_quantity(self) -> float:
        return float(sum(self.deliveries.values()))

    @property
    def total_cost(self) -> float:
        return float(self.distance_cost + self.time_cost)

#  heuristic class

class IRPHeuristic:
    def __init__(
        self,
        data_dir: str,
        suffix: Optional[str] = None,
        output_dir: str = ".",
        coverage_hours: int = DEFAULT_COVERAGE_HOURS,
    ):
        self.data_dir       = data_dir
        self.suffix         = str(suffix) if suffix is not None else None
        self.output_dir     = output_dir
        self.coverage_hours = int(coverage_hours)

        self.locations       = self._read_csv("locations")
        self.forecasts       = self._read_csv("forecasts")
        self.trailers        = self._read_csv("trailers")
        self.drivers         = self._read_csv("drivers")
        self.driver_windows  = self._read_csv("driver_windows")
        self.allowed_trailers = self._read_csv("allowed_trailers")
        self.time_matrix     = self._read_matrix("time_matrix")
        self.distance_matrix = self._read_matrix("distance_matrix")

        self._prepare_instance()



    def _file_path(self, base: str) -> str:
        data_path  = Path(self.data_dir)
        candidates = []
        if self.suffix is not None:
            candidates.append(data_path / f"{base}({self.suffix}).csv")
            candidates.append(data_path / f"{base}_{self.suffix}.csv")
        candidates.append(data_path / f"{base}.csv")
        candidates.extend(sorted(data_path.glob(f"{base}(*)*.csv")))
        candidates.extend(sorted(data_path.glob(f"{base}_*.csv")))

        seen, unique = set(), []
        for p in candidates:
            p = Path(p)
            if p not in seen:
                unique.append(p)
                seen.add(p)
        for p in unique:
            if p.exists():
                return str(p)
        raise FileNotFoundError(
            f"Cannot find '{base}' in {self.data_dir}. Tried: {[str(p) for p in unique]}"
        )

    def _read_csv(self, base: str) -> pd.DataFrame:
        return pd.read_csv(self._file_path(base))

    def _read_matrix(self, base: str) -> np.ndarray:
        return pd.read_csv(self._file_path(base), header=None).values.astype(float)



    def _prepare_instance(self) -> None:
        locs = self.locations.copy()
        locs["index"] = locs["index"].astype(int)
        locs = locs.sort_values("index").reset_index(drop=True)
        self.locations = locs

        self.location_indices = locs["index"].tolist()
        self.loc_to_pos       = {loc: pos for pos, loc in enumerate(self.location_indices)}

        n = len(self.location_indices)
        if self.time_matrix.shape[0] != n:
            raise ValueError(f"time_matrix shape {self.time_matrix.shape} != {n} locations.")
        if self.distance_matrix.shape[0] != n:
            raise ValueError(f"distance_matrix shape {self.distance_matrix.shape} != {n} locations.")

        base_rows     = locs[locs["type"].str.lower() == "base"]
        source_rows   = locs[locs["type"].str.lower() == "source"]
        customer_rows = locs[locs["type"].str.lower() == "customer"]

        if base_rows.empty or source_rows.empty or customer_rows.empty:
            raise ValueError("locations.csv must have at least one Base, Source, and Customer.")

        self.base     = int(base_rows.iloc[0]["index"])
        self.source   = int(source_rows.iloc[0]["index"])
        self.customers = [int(x) for x in customer_rows["index"].tolist()]

        self.setup_time = {int(r["index"]): float(r["setupTime"]) for _, r in locs.iterrows()}
        self.capacity   = {int(r["index"]): float(r["capacity"])  for _, r in customer_rows.iterrows()}
        self.safety     = {int(r["index"]): float(r["safety_level"]) for _, r in customer_rows.iterrows()}
        self.initial_inventory = {
            int(r["index"]): float(r["initial_tank_quantity"]) for _, r in customer_rows.iterrows()
        }

        demand_cols = sorted(
            [c for c in self.forecasts.columns if str(c).startswith("t")],
            key=lambda x: int(str(x)[1:]),
        )
        self.horizon_hours = len(demand_cols)
        self.demand: Dict[int, np.ndarray] = {}
        for _, row in self.forecasts.iterrows():
            loc = int(row["location_index"])
            self.demand[loc] = row[demand_cols].to_numpy(dtype=float)
        for c in self.customers:
            if c not in self.demand:
                self.demand[c] = np.zeros(self.horizon_hours)

        self.demand_cum = {
            c: np.concatenate(([0.0], np.cumsum(self.demand[c])))
            for c in self.customers
        }

        self.trailer_capacity      = {int(r["index"]): float(r["capacity"])      for _, r in self.trailers.iterrows()}
        self.trailer_distance_cost = {int(r["index"]): float(r["distance_cost"]) for _, r in self.trailers.iterrows()}

        self.driver_max_drive  = {int(r["index"]): float(r["max_driving_duration"])   for _, r in self.drivers.iterrows()}
        self.driver_trailer    = {int(r["index"]): int(r["trailer_index"])             for _, r in self.drivers.iterrows()}
        self.driver_rest       = {int(r["index"]): float(r["min_inter_shift_duration"]) for _, r in self.drivers.iterrows()}
        self.driver_time_cost  = {int(r["index"]): float(r["time_cost"])              for _, r in self.drivers.iterrows()}

        self.allowed: Set[Tuple[int, int]] = set(
            (int(r["location_index"]), int(r["trailer_index"]))
            for _, r in self.allowed_trailers.iterrows()
        )

        max_loc = max(self.location_indices)
        self.delivered    = np.zeros((max_loc + 1, self.horizon_hours), dtype=float)
        self.routes: List[RouteResult] = []
        self.next_available = {int(d): 0.0 for d in self.drivers["index"].tolist()}



    def dist(self, i: int, j: int) -> float:
        return float(self.distance_matrix[self.loc_to_pos[i], self.loc_to_pos[j]])

    def travel_time(self, i: int, j: int) -> float:
        return float(self.time_matrix[self.loc_to_pos[i], self.loc_to_pos[j]])

    def inv_at_start_of_hour(self, c: int, h: int) -> float:
        """Inventory at the start of hour h (before any deliveries or demand in hour h)."""
        h = int(max(0, min(h, self.horizon_hours)))
        return (
            self.initial_inventory[c]
            + float(self.delivered[c, :h].sum())
            - float(self.demand_cum[c][h])
        )

    def inv_before_new_delivery(self, c: int, h: int) -> float:
        """
        Inventory just before a new delivery in hour h, after accounting for
        deliveries already committed to hour h by earlier routes.
        """
        h = int(max(0, min(h, self.horizon_hours - 1)))
        return self.inv_at_start_of_hour(c, h) + float(self.delivered[c, h])



    def max_deliverable_without_overflow(self, c: int, h: int, cap_remaining: float) -> float:


        if h >= self.horizon_hours:
            return 0.0


        room_now = self.capacity[c] - self.inv_before_new_delivery(c, h)
        q = min(cap_remaining, max(0.0, room_now))
        if q < EPS:
            return 0.0


        cur = self.inv_at_start_of_hour(c, h)
        max_level = -1e18
        for t in range(h, self.horizon_hours):
            cur += float(self.delivered[c, t])
            if t == h:
                cur += q
            max_level = max(max_level, cur)
            cur -= float(self.demand[c][t])

        if max_level > self.capacity[c] - 1e-3:

            q -= (max_level - self.capacity[c]) + 0.01
            q = max(0.0, q)

        return q

    def hours_until_below(self, c: int, from_hour: int, threshold: float) -> float:

        from_hour = int(max(0, min(from_hour, self.horizon_hours - 1)))
        cur = self.inv_at_start_of_hour(c, from_hour)
        if cur < threshold:
            return 0.0
        for t in range(from_hour, self.horizon_hours):
            cur += float(self.delivered[c, t])
            cur -= float(self.demand[c][t])
            if cur < threshold:
                return float(t + 1 - from_hour)
        return float(self.horizon_hours + 1 - from_hour)

    def _target_quantity_cov(self, c: int, arrival_hour: int, remaining: float,
                              trailer_cap: float, coverage: int) -> float:

        if arrival_hour >= self.horizon_hours:
            return 0.0

        end     = min(self.horizon_hours, arrival_hour + coverage)
        inv_now = self.inv_before_new_delivery(c, arrival_hour)
        upcoming = float(self.demand[c][arrival_hour:end].sum())
        needed   = max(0.0, upcoming + self.safety[c] - inv_now)
        room     = self.capacity[c] - inv_now

        if room <= 0 or remaining <= 0:
            return 0.0

        if needed <= 0:
            q = min(remaining, room)
        else:
            q = min(remaining, room, needed)
            if remaining > 0.5 * trailer_cap:
                q = min(remaining, room)

        return self.max_deliverable_without_overflow(c, arrival_hour, q)

    def target_quantity(self, c: int, arrival_hour: int, remaining: float, trailer_cap: float) -> float:
        """Desired delivery quantity using the default coverage window."""
        return self._target_quantity_cov(c, arrival_hour, remaining, trailer_cap, self.coverage_hours)

    def useful_threshold(self, c: int, trailer_cap: float) -> float:

        return min(max(MIN_DELIVERY_ABS, MIN_DELIVERY_FRAC * trailer_cap), 0.20 * self.capacity[c])



    def urgency_level(self, c: int, eval_hour: int) -> Tuple[int, float, float]:

        stockout_h = self.hours_until_below(c, eval_hour, 0.0)
        safety_h   = self.hours_until_below(c, eval_hour, self.safety[c])

        if stockout_h <= 24:
            urgency = 4
        elif safety_h <= 24:
            urgency = 3
        elif safety_h <= 72:
            urgency = 2
        elif safety_h <= 168:
            urgency = 1
        else:
            urgency = 0
        return urgency, stockout_h, safety_h



    def compute_route_metrics(self, sequence: List[int], start_time: float) -> RouteMetrics:

        time       = float(start_time)
        total_dist = 0.0
        drive_time = 0.0
        arrivals_min: Dict[int, float] = {}
        arrivals_hour: Dict[int, int]  = {}

        for a, b in zip(sequence[:-1], sequence[1:]):
            tt = self.travel_time(a, b)
            dd = self.dist(a, b)
            time      += tt
            drive_time += tt
            total_dist += dd

            if b in self.customers:
                arrivals_min[b]  = time
                arrivals_hour[b] = int(max(0, min(time // 60, self.horizon_hours - 1)))

            if b != self.base:
                time += self.setup_time.get(b, 0.0)

        return RouteMetrics(
            sequence=list(sequence),
            arrivals_min=arrivals_min,
            arrivals_hour=arrivals_hour,
            distance=total_dist,
            drive_time=drive_time,
            duration=time - float(start_time),
            end_time=time,
        )

    def feasible_timing(self, metrics: RouteMetrics, window_end: float, max_drive: float) -> bool:
        return metrics.end_time <= window_end + EPS and metrics.drive_time <= max_drive + EPS

    def best_insertion_position(
        self,
        sequence: List[int],
        c: int,
        start_time: float,
        window_end: float,
        max_drive: float,
    ) -> Optional[Tuple[int, RouteMetrics, float]]:

        base_metrics = self.compute_route_metrics(sequence, start_time)
        best = None
        for p in range(2, len(sequence)):
            trial   = sequence[:p] + [c] + sequence[p:]
            metrics = self.compute_route_metrics(trial, start_time)
            if not self.feasible_timing(metrics, window_end, max_drive):
                continue
            extra = metrics.distance - base_metrics.distance
            if best is None or extra < best[2] - EPS:
                best = (p, metrics, extra)
        return best



    def assign_quantities_in_order(
        self,
        sequence: List[int],
        start_time: float,
        trailer_cap: float,
        window_end: float,
        max_drive: float,
    ) -> Tuple[List[int], Dict[int, float]]:

        customers_in_seq = [n for n in sequence if n in self.customers]
        if not customers_in_seq:
            return sequence, {}


        metrics_now = self.compute_route_metrics(sequence, start_time)
        def sort_key(c):
            h_arr = metrics_now.arrivals_hour.get(c, int(start_time // 60))
            urg, so_h, _ = self.urgency_level(c, h_arr)
            return (-urg, so_h)

        sorted_customers = sorted(customers_in_seq, key=sort_key)


        sequence = [self.base, self.source] + sorted_customers + [self.base]


        reordered_metrics = self.compute_route_metrics(sequence, start_time)
        if not self.feasible_timing(reordered_metrics, window_end, max_drive):

            sequence = [self.base, self.source] + customers_in_seq + [self.base]

        for _attempt in range(4):
            metrics = self.compute_route_metrics(sequence, start_time)
            if not self.feasible_timing(metrics, window_end, max_drive):
                return sequence, {}

            remaining  = float(trailer_cap)
            deliveries: Dict[int, float] = {}
            remove: Set[int] = set()

            for node in sequence:
                if node not in self.customers:
                    continue
                h         = metrics.arrivals_hour[node]
                urgency, _, _ = self.urgency_level(node, h)

                cov = self.horizon_hours if urgency >= 4 else self.coverage_hours
                q   = min(self._target_quantity_cov(node, h, remaining, trailer_cap, cov), remaining)
                q   = self.max_deliverable_without_overflow(node, h, q)
                threshold = self.useful_threshold(node, trailer_cap)

                if q > 0 and (urgency >= 4 or q >= threshold):
                    deliveries[node] = q
                    remaining       -= q
                elif urgency < 4:
                    remove.add(node)


            if not remove:
                return sequence, deliveries


            sequence = [n for n in sequence if n not in remove]
            customers_in_seq = [n for n in sequence if n in self.customers]
            if not customers_in_seq:
                return sequence, {}


        metrics   = self.compute_route_metrics(sequence, start_time)
        remaining = float(trailer_cap)
        deliveries = {}
        for node in sequence:
            if node not in self.customers:
                continue
            h         = metrics.arrivals_hour[node]
            urgency, _, _ = self.urgency_level(node, h)
            cov = self.horizon_hours if urgency >= 4 else self.coverage_hours
            q   = min(self._target_quantity_cov(node, h, remaining, trailer_cap, cov), remaining)
            q   = self.max_deliverable_without_overflow(node, h, q)
            if q > 0:
                deliveries[node] = q
                remaining       -= q
        return sequence, deliveries



    def two_opt(
        self,
        sequence: List[int],
        start_time: float,
        window_end: float,
        max_drive: float,
    ) -> List[int]:

        best         = list(sequence)
        best_metrics = self.compute_route_metrics(best, start_time)

        for _ in range(TWO_OPT_SWEEPS):
            improved = False
            n        = len(best)
            for i in range(2, n - 2):
                for j in range(i + 1, n - 1):
                    trial   = best[:i] + list(reversed(best[i: j + 1])) + best[j + 1:]
                    metrics = self.compute_route_metrics(trial, start_time)
                    if not self.feasible_timing(metrics, window_end, max_drive):
                        continue
                    if metrics.distance < best_metrics.distance - EPS:
                        best         = trial
                        best_metrics = metrics
                        improved     = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best



    def build_route(
        self,
        route_id: int,
        driver: int,
        window_start: float,
        window_end: float,
    ) -> Optional[RouteResult]:

        trailer        = self.driver_trailer[driver]
        trailer_cap    = self.trailer_capacity[trailer]
        dist_cost_rate = self.trailer_distance_cost[trailer]
        time_cost_rate = self.driver_time_cost[driver]
        max_drive      = self.driver_max_drive[driver]

        sequence: List[int]              = [self.base, self.source, self.base]
        planned_deliveries: Dict[int, float] = {}
        remaining = float(trailer_cap)
        min_remaining = max(MIN_DELIVERY_ABS, 0.02 * trailer_cap)


        proactive = self._proactive_must_visit(
            window_start, window_end, trailer, sequence, max_drive
        )
        for c in proactive:
            if c in planned_deliveries or remaining <= 0:
                continue
            ins = self.best_insertion_position(sequence, c, window_start, window_end, max_drive)
            if ins is None:
                continue
            p, ins_metrics, _ = ins
            arrival_h = ins_metrics.arrivals_hour[c]
            q = self._target_quantity_cov(c, arrival_h, remaining, trailer_cap, self.horizon_hours)
            if q <= 0:
                continue
            sequence = sequence[:p] + [c] + sequence[p:]
            planned_deliveries[c] = q
            remaining -= q

        while len(planned_deliveries) < MAX_CUSTOMERS_PER_ROUTE:

            if remaining <= min_remaining:

                has_critical = False
                for _c in self.customers:
                    if _c in planned_deliveries or (_c, trailer) not in self.allowed:
                        continue
                    _ins = self.best_insertion_position(sequence, _c, window_start, window_end, max_drive)
                    if _ins is None:
                        continue
                    _arr_h = _ins[1].arrivals_hour[_c]
                    if self.urgency_level(_c, _arr_h)[0] >= 4:
                        has_critical = True
                        break
                if not has_critical:
                    break
            best_candidate = None

            for c in self.customers:
                if c in planned_deliveries:
                    continue
                if (c, trailer) not in self.allowed:
                    continue

                insertion = self.best_insertion_position(
                    sequence, c, window_start, window_end, max_drive
                )
                if insertion is None:
                    continue

                p, metrics, extra_dist = insertion
                arrival_h = metrics.arrivals_hour[c]


                urgency, stockout_h, _ = self.urgency_level(c, arrival_h)


                cov = self.horizon_hours if urgency >= 4 else self.coverage_hours
                q = self._target_quantity_cov(c, arrival_h, remaining, trailer_cap, cov)

                threshold = self.useful_threshold(c, trailer_cap)

                if urgency >= 4:
                    if q <= 0:
                        continue
                else:
                    if q < threshold:
                        continue

                extra_cost = max(EPS, extra_dist * dist_cost_rate)
                efficiency = q / (extra_cost + 1.0)
                tie_break  = 10_000.0 / (stockout_h + 1.0)
                score      = URGENCY_WEIGHT * urgency + efficiency + tie_break

                if best_candidate is None or score > best_candidate["score"]:
                    best_candidate = {
                        "customer":  c,
                        "position":  p,
                        "quantity":  q,
                        "score":     score,
                    }

            if best_candidate is None:
                break

            c = int(best_candidate["customer"])
            p = int(best_candidate["position"])
            q = float(best_candidate["quantity"])
            sequence = sequence[:p] + [c] + sequence[p:]
            planned_deliveries[c] = q
            remaining -= q

        if not planned_deliveries:
            return None


        sequence = self.two_opt(sequence, window_start, window_end, max_drive)


        sequence, final_deliveries = self.assign_quantities_in_order(
            sequence, window_start, trailer_cap, window_end, max_drive
        )
        if not final_deliveries:
            return None


        metrics = self.compute_route_metrics(sequence, window_start)
        if not self.feasible_timing(metrics, window_end, max_drive):
            return None


        for c, q in final_deliveries.items():
            h = metrics.arrivals_hour[c]
            if 0 <= h < self.horizon_hours:
                self.delivered[c, h] += q

        return RouteResult(
            route_id=route_id,
            driver=driver,
            trailer=trailer,
            window_start=window_start,
            window_end=window_end,
            actual_start=window_start,
            sequence=sequence,
            deliveries=final_deliveries,
            arrivals_min=metrics.arrivals_min,
            arrivals_hour=metrics.arrivals_hour,
            distance=metrics.distance,
            drive_time=metrics.drive_time,
            duration=metrics.duration,
            end_time=metrics.end_time,
            distance_cost=metrics.distance * dist_cost_rate,
            time_cost=metrics.duration  * time_cost_rate,
        )



    def _next_window_arrival_hour(self, window_start: float, trailer: int, customer: int) -> Optional[int]:

        for _, w in self._sorted_windows.iterrows():
            start = float(w["start"])
            end   = float(w["end"])
            drv   = int(w["driver_index"])
            if start <= window_start:
                continue
            if self.driver_trailer[drv] != trailer:
                continue
            actual = max(start, self.next_available.get(drv, 0.0))
            if actual >= end - EPS:
                continue

            arr = actual + self.travel_time(self.base, self.source) +                   self.setup_time.get(self.source, 0.0) +                   self.travel_time(self.source, customer)
            h = int(arr // 60)
            if h < self.horizon_hours:
                return h
        return None

    def _proactive_must_visit(
        self,
        window_start: float,
        window_end: float,
        trailer: int,
        current_sequence: List[int],
        max_drive: float,
    ) -> List[int]:

        must_visit = []
        window_hour = int(max(0, window_start // 60))

        for c in self.customers:
            if (c, trailer) not in self.allowed:
                continue


            trial_seq = [self.base, self.source, c, self.base]
            metrics   = self.compute_route_metrics(trial_seq, window_start)
            if not self.feasible_timing(metrics, window_end, max_drive):
                continue  # can't reach c this window at all
            this_arrival_h = metrics.arrivals_hour[c]


            next_h = self._next_window_arrival_hour(window_start, trailer, c)

            if next_h is None:

                so_h = self.hours_until_below(c, this_arrival_h, 0.0)
                if so_h < (self.horizon_hours - this_arrival_h):
                    must_visit.append(c)
                continue


            inv_at_next = self.inv_at_start_of_hour(c, next_h)
            if inv_at_next < 0:
                must_visit.append(c)

        return must_visit



    def build_rescue_route(
        self,
        route_id: int,
        driver: int,
        window_start: float,
        window_end: float,
        seed_customer: int,
    ) -> Optional[RouteResult]:

        trailer        = self.driver_trailer[driver]
        trailer_cap    = self.trailer_capacity[trailer]
        dist_cost_rate = self.trailer_distance_cost[trailer]
        time_cost_rate = self.driver_time_cost[driver]
        max_drive      = self.driver_max_drive[driver]


        sequence = [self.base, self.source, seed_customer, self.base]
        metrics  = self.compute_route_metrics(sequence, window_start)
        if not self.feasible_timing(metrics, window_end, max_drive):
            return None

        arrival_h = metrics.arrivals_hour[seed_customer]
        q_seed = self._target_quantity_cov(
            seed_customer, arrival_h, trailer_cap, trailer_cap, self.horizon_hours
        )
        if q_seed <= 0:
            return None

        planned_deliveries: Dict[int, float] = {seed_customer: q_seed}
        remaining = trailer_cap - q_seed
        min_remaining = max(MIN_DELIVERY_ABS, 0.02 * trailer_cap)


        while remaining > min_remaining and len(planned_deliveries) < MAX_CUSTOMERS_PER_ROUTE:
            best_candidate = None
            for c in self.customers:
                if c in planned_deliveries:
                    continue
                if (c, trailer) not in self.allowed:
                    continue
                insertion = self.best_insertion_position(
                    sequence, c, window_start, window_end, max_drive
                )
                if insertion is None:
                    continue
                p, ins_metrics, extra_dist = insertion
                arrival_h = ins_metrics.arrivals_hour[c]
                urgency, stockout_h, _ = self.urgency_level(c, arrival_h)
                # Consistent with build_route: use full-horizon coverage for urgency>=4
                cov = self.horizon_hours if urgency >= 4 else self.coverage_hours
                q = self._target_quantity_cov(c, arrival_h, remaining, trailer_cap, cov)
                threshold = self.useful_threshold(c, trailer_cap)
                if urgency >= 4:
                    if q <= 0:
                        continue
                else:
                    if q < threshold:
                        continue
                extra_cost = max(EPS, extra_dist * dist_cost_rate)
                tie_break  = 10_000.0 / (stockout_h + 1.0)
                score = URGENCY_WEIGHT * urgency + q / (extra_cost + 1.0) + tie_break
                if best_candidate is None or score > best_candidate["score"]:
                    best_candidate = {"customer": c, "position": p, "quantity": q, "score": score}

            if best_candidate is None:
                break
            c = int(best_candidate["customer"])
            p = int(best_candidate["position"])
            q = float(best_candidate["quantity"])
            sequence = sequence[:p] + [c] + sequence[p:]
            planned_deliveries[c] = q
            remaining -= q

        sequence = self.two_opt(sequence, window_start, window_end, max_drive)
        sequence, final_deliveries = self.assign_quantities_in_order(
            sequence, window_start, trailer_cap, window_end, max_drive
        )
        if not final_deliveries:
            return None

        metrics = self.compute_route_metrics(sequence, window_start)
        if not self.feasible_timing(metrics, window_end, max_drive):
            return None

        for c, q in final_deliveries.items():
            h = metrics.arrivals_hour[c]
            if 0 <= h < self.horizon_hours:
                self.delivered[c, h] += q

        return RouteResult(
            route_id=route_id,
            driver=driver,
            trailer=trailer,
            window_start=window_start,
            window_end=window_end,
            actual_start=window_start,
            sequence=sequence,
            deliveries=final_deliveries,
            arrivals_min=metrics.arrivals_min,
            arrivals_hour=metrics.arrivals_hour,
            distance=metrics.distance,
            drive_time=metrics.drive_time,
            duration=metrics.duration,
            end_time=metrics.end_time,
            distance_cost=metrics.distance * dist_cost_rate,
            time_cost=metrics.duration  * time_cost_rate,
        )



    def solve(self) -> Dict[str, float]:
        windows = self.driver_windows.copy()
        windows["driver_index"] = windows["driver_index"].astype(int)
        windows = windows.sort_values(["start", "driver_index"]).reset_index(drop=True)

        self._sorted_windows = windows

        route_id = 0
        for _, w in windows.iterrows():
            driver = int(w["driver_index"])
            start  = float(w["start"])
            end    = float(w["end"])


            actual_start = max(start, self.next_available.get(driver, 0.0))
            if actual_start >= end - EPS:
                continue
            if int(actual_start // 60) >= self.horizon_hours:
                continue

            route = self.build_route(route_id, driver, actual_start, end)
            if route is None:
                continue

            self.routes.append(route)
            self.next_available[driver] = route.end_time + self.driver_rest[driver]
            route_id += 1


        for _rescue_round in range(3):
            at_risk = self._customers_with_stockout()
            if not at_risk:
                break

            made_progress = False
            for c_rescue in at_risk:

                for _, w in windows.iterrows():
                    driver = int(w["driver_index"])
                    trailer_k = self.driver_trailer[driver]
                    if (c_rescue, trailer_k) not in self.allowed:
                        continue

                    start = float(w["start"])
                    end   = float(w["end"])
                    actual_start = max(start, self.next_available.get(driver, 0.0))
                    if actual_start >= end - EPS:
                        continue
                    if int(actual_start // 60) >= self.horizon_hours:
                        continue


                    window_hour = int(actual_start // 60)
                    stockout_h  = self.hours_until_below(c_rescue, window_hour, 0.0)
                    if stockout_h <= 0:
                        continue

                    route = self.build_rescue_route(
                        route_id, driver, actual_start, end, c_rescue
                    )
                    if route is None:
                        continue

                    self.routes.append(route)
                    self.next_available[driver] = route.end_time + self.driver_rest[driver]
                    route_id += 1
                    made_progress = True
                    break

            if not made_progress:
                break

        return self.compute_metrics()

    def _customers_with_stockout(self) -> List[int]:

        at_risk = []
        for c in self.customers:
            cur = float(self.initial_inventory[c])
            for h in range(self.horizon_hours):
                cur += float(self.delivered[c, h])
                cur -= float(self.demand[c][h])
                if cur < -EPS:
                    at_risk.append(c)
                    break
        return at_risk



    def simulate_inventory(self) -> pd.DataFrame:
        rows = []
        for c in self.customers:
            cur = float(self.initial_inventory[c])
            inv_values = []
            for h in range(self.horizon_hours):
                cur += float(self.delivered[c, h])
                cur -= float(self.demand[c][h])
                inv_values.append(cur)
            row = {"location_index": c}
            row.update({f"t{h}": inv_values[h] for h in range(self.horizon_hours)})
            rows.append(row)
        return pd.DataFrame(rows)



    def compute_metrics(self) -> Dict[str, float]:
        total_distance = sum(r.distance for r in self.routes)
        total_duration = sum(r.duration for r in self.routes)
        distance_cost  = sum(r.distance_cost for r in self.routes)
        time_cost      = sum(r.time_cost for r in self.routes)
        total_cost     = distance_cost + time_cost
        total_quantity = sum(r.total_quantity for r in self.routes)
        lr = total_cost / total_quantity if total_quantity > 0 else float("inf")

        stockout_hours         = 0
        safety_violation_hours = 0
        overflow_hours         = 0
        final_inventory_sum    = 0.0

        for c in self.customers:
            cur = float(self.initial_inventory[c])
            for h in range(self.horizon_hours):
                cur += float(self.delivered[c, h])
                if cur > self.capacity[c] + 1e-3:  # 1e-3 avoids float rounding false positives
                    overflow_hours += 1
                cur -= float(self.demand[c][h])
                if cur < -EPS:
                    stockout_hours += 1
                if cur < self.safety[c] - EPS:
                    safety_violation_hours += 1
            final_inventory_sum += cur

        return {
            "suffix":                  self.suffix if self.suffix is not None else "none",
            "horizon_hours":           self.horizon_hours,
            "num_customers":           len(self.customers),
            "num_routes":              len(self.routes),
            "total_distance":          total_distance,
            "total_duration_minutes":  total_duration,
            "distance_cost":           distance_cost,
            "time_cost":               time_cost,
            "total_cost":              total_cost,
            "total_quantity_delivered": total_quantity,
            "LR":                      lr,
            "stockout_hours":          stockout_hours,
            "safety_violation_hours":  safety_violation_hours,
            "tank_overflow_hours":     overflow_hours,
            "final_inventory_sum":     final_inventory_sum,
        }



    def save_outputs(self, metrics: Dict[str, float]) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        tag = self.suffix if self.suffix is not None else "run"

        pd.DataFrame([metrics]).to_csv(
            os.path.join(self.output_dir, f"metrics_{tag}.csv"), index=False
        )

        route_rows, delivery_rows = [], []
        for r in self.routes:
            route_rows.append({
                "route_id":    r.route_id,
                "driver":      r.driver,
                "trailer":     r.trailer,
                "window_start": r.window_start,
                "window_end":  r.window_end,
                "actual_start": r.actual_start,
                "end_time":    r.end_time,
                "sequence":    "-".join(map(str, r.sequence)),
                "num_customers": len(r.deliveries),
                "quantity":    r.total_quantity,
                "distance":    r.distance,
                "drive_time":  r.drive_time,
                "duration":    r.duration,
                "distance_cost": r.distance_cost,
                "time_cost":   r.time_cost,
                "total_cost":  r.total_cost,
            })
            for c, q in r.deliveries.items():
                delivery_rows.append({
                    "route_id":    r.route_id,
                    "driver":      r.driver,
                    "trailer":     r.trailer,
                    "customer":    c,
                    "arrival_min": r.arrivals_min[c],
                    "arrival_hour": r.arrivals_hour[c],
                    "quantity":    q,
                })

        pd.DataFrame(route_rows).to_csv(
            os.path.join(self.output_dir, f"routes_{tag}.csv"), index=False
        )
        pd.DataFrame(delivery_rows).to_csv(
            os.path.join(self.output_dir, f"deliveries_{tag}.csv"), index=False
        )
        self.simulate_inventory().to_csv(
            os.path.join(self.output_dir, f"inventory_{tag}.csv"), index=False
        )




def run_group_folder(
    group_folder: str | Path,
    output_root: str | Path = "outputs",
    coverage_hours: int = DEFAULT_COVERAGE_HOURS,
) -> Dict[str, float]:
    group_folder = Path(group_folder)
    output_dir   = Path(output_root) / group_folder.name
    solver = IRPHeuristic(
        data_dir=str(group_folder),
        suffix=None,
        output_dir=str(output_dir),
        coverage_hours=coverage_hours,
    )
    metrics = solver.solve()
    metrics["group"] = group_folder.name
    solver.save_outputs(metrics)
    return metrics


def run_all_groups(
    data_root: str | Path = ".",
    group_names: List[str] = [f"Instance_V_1.{i}" for i in range(1, 12)],
    output_root: str | Path = "outputs",
    coverage_hours: int = DEFAULT_COVERAGE_HOURS,
) -> pd.DataFrame:
    data_root   = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    t_total_start = time.time()
    for name in group_names:
        group_path = data_root / name
        if not group_path.exists():
            print(f"Skipping {name}: not found at {group_path}")
            continue
        print(f"Running {name} ...", flush=True)
        t_instance = time.time()
        try:
            m = run_group_folder(group_path, output_root=output_root, coverage_hours=coverage_hours)
            elapsed = time.time() - t_instance
            all_metrics.append(m)
            print(
                f"  routes={m['num_routes']:3d}  "
                f"qty={m['total_quantity_delivered']:,.0f}  "
                f"cost={m['total_cost']:,.0f}  "
                f"LR={m['LR']:.6f}  "
                f"stockout_h={m['stockout_hours']}  "
                f"overflow_h={m['tank_overflow_hours']}  "
                f"safety_viol_h={m['safety_violation_hours']}  "
                f"time={elapsed:.1f}s"
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")

    total_elapsed = time.time() - t_total_start
    print(f"\nTotal runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    df = pd.DataFrame(all_metrics)
    if not df.empty:
        out_path = output_root / "metrics_all_groups.csv"
        df.to_csv(out_path, index=False)
        print(f"\nAll-group metrics saved to: {out_path}")
    return df


def _normalize_group_names(values: List[str]) -> List[str]:

    result = []
    for v in values:
        t = str(v).strip()
        if t.lower().startswith("instance_"):
            result.append(t)
        else:
            # Bare number -> Instance_V_1.<number>
            result.append(f"Instance_V_1.{t}")
    return result



def main() -> None:
    parser = argparse.ArgumentParser(
        description="IRP heuristic v2 — run on Instance_V_1.1 … Instance_V_1.11 folders."
    )
    parser.add_argument("--data-root",      default=".",       help="Parent directory of instance folders.")
    parser.add_argument("--output-root",    default="outputs", help="Output directory.")
    parser.add_argument("--groups", nargs="+", default=[str(i) for i in range(1, 12)],
                        help="Instances to run, e.g. 1 2 3 or Instance_V_1.1 Instance_V_1.2.")
    parser.add_argument("--coverage-hours", type=int, default=DEFAULT_COVERAGE_HOURS,
                        help=f"Look-ahead for delivery quantity (default {DEFAULT_COVERAGE_HOURS}).")
    parser.add_argument("--inspect-group",  default=None,
                        help="Print first rows of route/delivery output for this instance (e.g. 1 or Instance_V_1.1).")
    args = parser.parse_args()

    group_names = _normalize_group_names(args.groups)
    df = run_all_groups(
        data_root=args.data_root,
        group_names=group_names,
        output_root=args.output_root,
        coverage_hours=args.coverage_hours,
    )

    if not df.empty:
        print("\nSummary:")
        print(df.to_string(index=False))

    if args.inspect_group:
        name = _normalize_group_names([args.inspect_group])[0]
        base = Path(args.output_root) / name
        for fname in (f"routes_run.csv", f"deliveries_run.csv"):
            p = base / fname
            if p.exists():
                print(f"\n{p}:")
                print(pd.read_csv(p).head().to_string(index=False))
            else:
                print(f"\nNot found: {p}")


if __name__ == "__main__":
    run_all_groups(data_root=".", output_root="outputs")
