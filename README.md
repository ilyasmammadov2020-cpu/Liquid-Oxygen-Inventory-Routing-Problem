# Liquid-Oxygen-Inventory-Routing-Problem
# Liquid Oxygen Inventory Routing Problem (LO-IRP)

## Overview

This repository contains a heuristic solution for the **Liquid Oxygen Inventory Routing Problem (LO-IRP)**, developed as part of an Industrial Engineering course project.

The project addresses a real-world logistics problem inspired by **Air Liquide**, a global supplier of industrial and medical gases. The objective is to optimize the distribution of liquid oxygen (LOX) over a **30-day planning horizon** while satisfying operational, inventory, and workforce constraints.

The problem combines several optimization domains:

- Vehicle Routing Problem (VRP)
- Inventory Routing Problem (IRP)
- Vendor Managed Inventory (VMI)
- Driver Scheduling
- Heterogeneous Fleet Optimization

---

## Problem Description

Air Liquide operates under a **Vendor Managed Inventory (VMI)** policy, meaning customers do not place replenishment orders. Instead, the supplier continuously monitors customer inventory levels and decides:

- When each customer should be visited
- How much liquid oxygen should be delivered
- Which vehicle and driver should perform each delivery

The distribution network consists of:

- **Bases** – starting and ending locations for drivers
- **Sources** – production plants where trailers are loaded
- **Customers** – demand locations equipped with cryogenic storage tanks

The goal is to generate a feasible delivery schedule that minimizes transportation cost while ensuring uninterrupted oxygen supply.

---

## Constraints

The heuristic considers several real-world operational constraints, including:

- Vehicle–customer compatibility
- Heterogeneous trailer capacities
- Driver working-hour regulations
- Mandatory driver rest periods
- Customer inventory limits
- Tank capacity restrictions
- Multi-stop delivery routes
- Shift duration limits
- Hourly customer demand forecasts

---

## Objective Function

The optimization objective is to minimize the **Logistics Ratio (LR)**:

\[
LR = \frac{\text{Total Operating Cost}}{\text{Total Quantity Delivered}}
\]

where

- **Operating Cost** includes:
  - Distance-based transportation costs
  - Driver wage costs

- **Delivered Quantity** represents the total amount of liquid oxygen supplied during the planning horizon.

This objective balances transportation efficiency and delivery effectiveness.

---

## Solution Approach

The project implements a heuristic optimization framework that includes:

- Rolling Horizon Planning
- Greedy Customer Selection
- Route Construction
- Inventory Feasibility Checks
- Driver Availability Validation
- Local Search Improvements
- Sensitivity Analysis

The heuristic produces feasible delivery schedules while satisfying operational constraints.

---

## Repository Structure

```
Liquid-Oxygen-Inventory-Routing-Problem
│
├── Instance_V_*
│   ├── customers.csv
│   ├── demand.csv
│   ├── vehicles.csv
│   └── ...
│
├── irp_heuristic.py
├── sensitivity_analysis.py
├── Case.pdf
├── README.md
└── LICENSE
```

---

## Requirements

- Python 3.10+
- pandas
- numpy
- matplotlib

Install dependencies using:

```bash
pip install pandas numpy matplotlib
```

---

## Running the Project

Run the heuristic:

```bash
python irp_heuristic.py
```

Run the sensitivity analysis:

```bash
python sensitivity_analysis.py
```

---

## Results

The implemented heuristic generates feasible delivery plans that satisfy inventory, routing, and driver constraints while minimizing the Logistics Ratio.

The repository also includes multiple benchmark instances for testing different scenarios and performing sensitivity analysis.

---

## Report

A detailed description of the mathematical model, heuristic algorithm, computational experiments, and results can be found in:

**Case.pdf**

---

## License

This project is licensed under the MIT License.


## Documentation

The complete technical report is not included in this repository.

If you are a recruiter, professor, or researcher and would like to review the full report, please feel free to contact me

