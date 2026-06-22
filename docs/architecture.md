# Territory Optimizer — Complete Architecture & Implementation Guide

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Layer](#3-data-layer)
4. [Feature Engineering](#4-feature-engineering)
5. [Core Optimization Algorithms](#5-core-optimization-algorithms)
6. [Post-Solve Enforcement](#6-post-solve-enforcement)
7. [API Layer](#7-api-layer)
8. [Web Dashboard](#8-web-dashboard)
9. [Database & Scheduler](#9-database--scheduler)
10. [Configuration Reference](#10-configuration-reference)
11. [CLI Reference](#11-cli-reference)
12. [Complete Change Log](#12-complete-change-log)

---

## 1. Project Overview

**Territory Optimizer** is an enterprise-grade system for rebalancing field territory assignments. It solves a large-scale combinatorial optimization problem:

- **Dealers** are individual sales points distributed across Indian metro areas.
- **FTCs** (Field Territory Coordinators) are field employees who manage dealer networks.
- The system takes an existing (suboptimal) assignment of dealers to FTCs and produces a new optimized assignment.

### Core Objectives

| Objective | Weight | Description |
|-----------|--------|-------------|
| Maximize compatibility | `alpha_1` (0.5) | Product match, historical performance, dealer type match |
| Minimize travel distance | `alpha_2` (0.3) | Geographical proximity using Haversine distance |
| Minimize disruption | `lambda` (0.1–5.0) | Keep dealers at their current FTC when beneficial |

### Hard Constraints

| Constraint | Default | Description |
|-----------|---------|-------------|
| Max dealers per FTC | 25 | No FTC can exceed this load |
| Min dealers per active FTC | 1 | Active FTCs must have at least one dealer |
| Max travel radius | 50 km | Dealer-FTC distance cannot exceed this |
| Workload balance threshold | 0.3 | Workload distribution constraint |

### Scale

- Designed for up to **100,000 dealers** and **30,000 FTCs**
- Pipeline completes in **~40 seconds** for production-size datasets
- Synthetic data generation supports 10 Indian metro areas

---

## 2. System Architecture

```
main.py
  ├── generate ──> SyntheticDataGenerator ──> *.parquet files
  ├── optimize ──> OptimizationPipeline.run()
  │                  ├─ DataLoader
  │                  ├─ DataValidator
  │                  ├─ DataProcessor
  │                  ├─ SpatialClusterer (K-Means)
  │                  ├─ Feature Engineers
  │                  │    ├─ WorkloadEngineer
  │                  │    ├─ CapacityEngineer
  │                  │    ├─ CompatibilityEngineer
  │                  │    └─ DistanceEngineer
  │                  ├─ TerritoryModel
  │                  │    ├─ OptimizationConstraints
  │                  │    └─ TerritorySolver
  │                  │         ├─ Greedy Cluster Solver (large)
  │                  │         └─ OR-Tools CP-SAT (small)
  │                  ├─ ResultAnalyzer
  │                  └─ Database (SQLite)
  └── serve ──> Flask API
                  ├─ REST endpoints
                  ├─ Static dashboard (Leaflet.js)
                  └─ APScheduler (daily cron)
```

### Pipeline Steps

The `OptimizationPipeline` executes 7 sequential steps:

| Step | Component | Description |
|------|-----------|-------------|
| 1 | DataLoader | Load dealers, FTCs, relationships, proximity from parquet |
| 2 | DataValidator | Validate data integrity (warnings don't stop pipeline) |
| 3 | DataProcessor | Build index mappings and assignment matrix |
| 4 | SpatialClusterer | Group nearby dealers into micro-location clusters |
| 5 | Feature Engineers | Compute workload, capacity, compatibility, distance matrices |
| 6 | TerritoryModel | Run optimization solver |
| 7 | ResultAnalyzer | Compute business impact and store results |

---

## 3. Data Layer

### 3.1 Source Files

All data is stored as Apache Parquet files in `data/`:

| File | Contents | Key Columns |
|------|----------|-------------|
| `dealers.parquet` | Individual sales points | `dealer_id`, `dealer_latitude`, `dealer_longitude`, `product_group`, `city` |
| `ftcs.parquet` | Territory coordinators | `ftc_id`, `ftc_latitude`, `ftc_longitude`, `product_group`, `city` |
| `relationships.parquet` | Current assignments | `dealer_id`, `ftc_id`, `product_category` |
| `proximity.parquet` | Dealer-to-dealer distances | `dealer_id`, `related_dealer_id`, `spatial_distance` |

### 3.2 Synthetic Data Generation

The `SyntheticDataGenerator` creates realistic test data with intentional inefficiencies:

**Dealer Distribution (10 Indian Metros)**

| City | Weight | Radius (deg) |
|------|--------|-------------|
| Mumbai | 20% | 0.20 |
| Pune | 15% | 0.15 |
| Bangalore | 15% | 0.18 |
| Delhi | 15% | 0.20 |
| Hyderabad | 10% | 0.15 |
| Chennai | 8% | 0.15 |
| Kolkata | 7% | 0.15 |
| Ahmedabad | 5% | 0.12 |
| Jaipur | 3% | 0.10 |
| Lucknow | 2% | 0.10 |

**Spatial Structure:**
- Each city has 3–8 sub-clusters to simulate neighborhoods
- Dealers placed with Gaussian noise (radius × 0.15) around sub-cluster centers
- FTCs anchored near random dealers within each city with 100–800 m jitter
- The jitter uses uniform random angle + radius (not Gaussian), ensuring FTCs never sit directly on top of a dealer

**Relationship Generation:**
- **85% same-city** assignment (preferring product-matched FTCs 70% of the time)
- **15% cross-city** misassignments (simulating real-world inefficiency)
- Every dealer gets exactly one relationship

### 3.3 Data Processing

The `DataProcessor` transforms raw data for optimization:

1. **Index Mapping** — Sorted string IDs → sequential integer indices
   - `D000001` → index 0, `D000002` → index 1, etc.
2. **Product Group Normalization** — Lowercase, strip whitespace
3. **Assignment Matrix** — Binary `[num_dealers × num_ftcs]` matrix from relationships

---

## 4. Feature Engineering

### 4.1 Workload (`WorkloadEngineer`)

Per-dealer workload score (MinMaxScaled to [0, 1]):

```
workload = 0.7 × normalized(cases_per_day) + 0.3 × normalized(disbursements)
```

### 4.2 Capacity (`CapacityEngineer`)

Per-FTC capacity score (MinMaxScaled to [0, 1]):

```
capacity = 0.4 × per_sum_mob + 0.3 × ntb_share + 0.2 × cross_sell + 0.1 × ftc_vintage
```

### 4.3 Compatibility (`CompatibilityEngineer`)

Per-dealer-FTC pair score [0, 1] in a `[D × F]` matrix:

```
compatibility = 0.6 × product_match + 0.3 × historical_performance + 0.1 × dealer_type_match
```

Where:
- **product_match**: 1 if dealer and FTC share any product group, else 0
- **historical_performance**: Normalized average cases per day from relationship history
- **dealer_type_match**: 1 if compatible (static dealer ↔ any FTC, mobile dealer ↔ high-mobility FTC)

### 4.4 Distance (`DistanceEngineer`)

Per-dealer-FTC Haversine distance in a `[D × F]` matrix:

```
distance_km = haversine(dealer_lat, dealer_lon, ftc_lat, ftc_lon)
distance = MinMaxScaler(distance_km)
```

Both raw km and normalized [0, 1] matrices are returned. The normalized version is used in the objective function; the raw km version is used for feasibility filtering.

---

## 5. Core Optimization Algorithms

### 5.1 Solver Strategy Selection

The `TerritorySolver` selects between two algorithms based on problem size:

| Condition | Solver Used | Complexity |
|-----------|-------------|------------|
| Clusters exist AND > 1 unique cluster | Greedy Cluster Solver | O(C² × F) |
| No clusters or single cluster | OR-Tools CP-SAT | O(D × F × scale) |

Where C = clusters, F = FTCs, D = dealers.

### 5.2 Greedy Cluster Solver (5 Phases)

#### Phase 1 — Initial Assignment

1. Group dealers into clusters (from `SpatialClusterer`)
2. Compute cluster-level mean distance to each FTC
3. Sort clusters by number of feasible FTCs (most constrained first)
4. Run 6 randomized trials with shuffling
5. For each cluster: assign to nearest feasible FTC with capacity
6. **If disruption minimization is active** (`lambda > 0.5`): prefer keeping cluster at its current FTC if feasible and has room
7. Score each trial: `-mean_distance + lambda × kept_ratio`
8. Select the trial with the highest score

#### Phase 2 — Repair Overloaded FTCs

1. Find FTCs exceeding `max_dealers`
2. For each overloaded FTC (sorted by overflow amount):
   - Sort assigned clusters by size (largest first)
   - Move clusters to the nearest feasible FTC with spare capacity
3. Up to 50 iterations; logs warning if any remain overloaded

#### Phase 3 — Activate Unused FTCs

1. For each unused FTC (sorted by total distance to eligible clusters):
   - Find the nearest cluster whose current FTC has multiple clusters assigned
   - Move that cluster to the unused FTC if capacity allows
2. Up to 20 iterations

#### Phase 4 — Swap Refinement

1. For every pair of FTCs (j, k):
   - For every cluster pair (c₁ in j, c₂ in k):
     - Check if swapping would reduce total mean distance
     - Check feasibility and capacity constraints
     - Execute swap if beneficial
2. Up to 20 iterations
3. **Skipped when disruption minimization is active** (would undo kept assignments)

#### Phase 5 — Final Repair + Activation

1. Fix any overloads created by swaps using one-cluster-at-a-time moves
2. Final pass to activate remaining unused FTCs
3. Convert cluster-level assignments back to dealer-level binary matrix

### 5.3 CP-SAT Solver

For small problems (no clusters), uses Google OR-Tools CP-SAT:

```
Variables:  x[i,j] ∈ {0, 1}  for each feasible dealer-FTC pair
Constraints:
  - Each dealer assigned to exactly one FTC
  - Each FTC's load ∈ [min_dealers, max_dealers] (or zero if inactive)

Objective (maximize):
  Σ x[i,j] × (alpha_1 × compat[i,j] + alpha_2 × (1 - dist[i,j]) + lambda × current[i,j])

Scaled by 1000 for integer arithmetic.
```

- Warm-started with current assignment hints
- Configurable time limit, workers, and relative gap limit

### 5.4 Feasibility Mask

The feasibility mask is a binary `[D × F]` matrix that restricts which FTCs each dealer can be assigned to:

1. **Distance Mask**: `distance_km[i,j] <= max_travel_radius_km`
2. **Product Compatibility Mask**: dealer and FTC share at least one product group
3. **Combined Mask**: `distance_mask AND product_mask`
4. **Fallback Repair**: Any dealer with zero feasible options gets their best `(compatibility - distance)` FTC force-assigned
5. **Cluster Cohesion Expansion**: Within each cluster, unionize the mask so all members share the same feasible FTC set

### 5.5 Objective Function (raw)

```
objective = Σ x[i,j] × (alpha_1 × compat[i,j] + alpha_2 × (1 - dist[i,j]))
            + lambda × Σ x[i,j] × current_assignment[i,j]
```

---

## 6. Post-Solve Enforcement

### 6.1 Problem

The cluster cohesion expansion (Phase 1 step 5 of the mask) allows a dealer to be assigned to an FTC that their individual feasibility mask blocks, as long as any cluster-mate can reach it. This creates cross-city assignments (e.g., a Mumbai dealer assigned to a Jaipur FTC because they share a cluster with a Mumbai dealer who can reach that FTC).

### 6.2 Enforcement Pass

After the solver returns, a two-pass enforcement loop runs:

**Pass 1 — Respect Capacity:**
```
for each dealer i:
    current_ftc = argmax(assignment[i])
    if pre_expansion_mask[i][current_ftc] == 1:
        skip (already feasible)
    find closest FTC in pre_expansion_mask with spare capacity
    move dealer there
```

**Pass 2 — Relax Capacity:**
```
for each remaining violating dealer i:
    find closest FTC in pre_expansion_mask (ignore capacity)
    move dealer there
```

**Metrics recomputed after enforcement:**
- `total_changes`, `changed_dealers`, `change_rate`
- `ftcs_used`
- `mean_distance_km`, `median_distance_km`

### 6.3 Remaining Violations

After enforcement, some dealers may still violate the raw `max_travel_radius_km` check. This happens when a dealer's closest feasible FTC (by the pre-expansion mask) is still beyond the radius. This is a data limitation — no FTC exists within the radius for that dealer. The enforcement gives the best possible assignment.

---

## 7. API Layer

### 7.1 Endpoints

All routes are under the `/api/v1` blueprint.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/optimize` | Run optimization with parameter overrides |
| GET | `/solution/<job_id>` | Get solution metadata |
| GET | `/solution/<job_id>/changes` | Get dealer-level changes |
| GET | `/scheduler/status` | Get scheduler status |
| POST | `/generate` | Generate synthetic data |
| GET | `/data/dealers` | Get all dealers |
| GET | `/data/ftcs` | Get all FTCs |
| GET | `/data/relationships` | Get original relationships |

### 7.2 Parameter Overrides

The `/optimize` endpoint accepts a `parameters` object. Supported keys:

| Key | Example | Description |
|-----|---------|-------------|
| `solver.time_limit_seconds` | `30` | Override solver time limit |
| `optimization.lambda` | `5.0` | Override disruption penalty |
| `constraints.max_travel_radius_km` | `100` | Override max distance |
| `constraints.max_dealers_per_ftc` | `30` | Override max load |

### 7.3 Data Flow

```
UI (POST /optimize)  →  API  →  OptimizationPipeline.run(parameters)
                                   ├─ _build_run_config(parameters)
                                   │    merges defaults with overrides
                                   └─ returns result dict

UI (GET /solution/<id>/changes)  →  API  →  Database
                                   returns [{dealer_id, from_ftc, to_ftc, impact_score}]
```

---

## 8. Web Dashboard

### 8.1 Stack

- **Map**: Leaflet.js with OpenStreetMap tiles (dark-themed via CSS filter)
- **Layout**: Sidebar (350 px) + Map area with overlay stats bar
- **Styling**: Dark theme, glass-morphism panels, CSS custom properties

### 8.2 Data Generation Panel

Controls:
- **Dealer count** — number input (100–100,000, default 1,000)
- **FTC count** — number input (10–30,000, default 100)
- **Generate Data** button → `POST /api/v1/generate`

### 8.3 Optimization Panel

Controls:
- **Minimize Disruption** checkbox → sends `optimization.lambda = 5.0`
- **Limit Dealer-FTC Distance** checkbox + number input (default 50 km):
  - Checked: sends the entered value
  - Unchecked: sends `999` (effectively unlimited)
- **Run Optimization** button → `POST /api/v1/optimize`

### 8.4 Visualization Layers

All toggles:

| Toggle | Layer | Description |
|--------|-------|-------------|
| Show FTCs | Green circles (12 px) at centroid of assigned dealers |
| Show Dealers | Blue dots (6 px), orange when unchanged after optimization |
| Old Assignments | Red semi-transparent lines from dealer to original FTC |
| New Territories | Purple lines from dealer to optimized FTC centroid |
| Show Distances | White km labels at midpoints (text-shadow only, no box) |
| Measure Distance | Crosshair cursor, numbered waypoints, running total |

### 8.5 Interactive Map Features

| Feature | Behavior |
|---------|----------|
| Click dealer/FTC | Highlights all connections to/from that entity |
| Popup | Shows dealer ID, original FTC, optimized FTC, changed indicator |
| Territory polygon | Dashed boundary for FTCs with 3–10 dealers |
| Distance pills | Haversine distance at midpoint of connections |
| Dealer colors | Blue = changed, Orange = unchanged |
| Lines colors | Purple = optimized, Red = original, Green (FTC→dealer) = unchanged dealer lines |

### 8.6 Stats Panel (Collapsible)

| Metric | Computation |
|--------|------------|
| Dealers Retained at Same FTC | `count` and `%` of dealers NOT in changes array |
| Dealers Changed FTC | `count` and `%` of dealers in changes array |
| Unallocated Dealers | Dealers with no FTC in final assignment |
| Active FTCs | FTCs with ≥1 dealer assigned |
| Unallocated / Idle FTCs | Total FTCs − Active FTCs |
| Min / Max / Mean / Median per Active FTC | Distribution of dealers across FTCs |

### 8.7 Map Overlay Stats Bar

| Stat | Source |
|------|--------|
| Total Dealers | `data.dealers.length` |
| Total FTCs | `data.ftcs.length` |
| Optimization Status | Set by `runOptimization()` response |

### 8.8 JavaScript Architecture

State object:
```javascript
let data = {
    dealers: [],       // dealer objects from API
    ftcs: [],          // FTC objects from API
    relationships: [], // original assignments
    changes: []        // optimized changes
};
```

Key functions:

| Function | Responsibility |
|----------|---------------|
| `fetchInitialData()` | Load dealers, FTCs, relationships on page load |
| `generateData()` | POST generate → clear state → re-fetch |
| `runOptimization()` | POST optimize → store job_id → fetch changes |
| `fetchChanges()` | GET solution changes → update data.changes → re-render |
| `renderMap()` | Draw FTCs, dealers, old links |
| `renderNewTerritories()` | Draw optimized purple links + dealer colors |
| `renderDistances()` | Draw distance pills (toggled) |
| `updateStatsPanel()` | Compute and display stats |
| `startMeasure()` / `stopMeasure()` | Distance measurement tool |
| `haversineKm()` | Geodesic distance calculation |
| `updateHighlighting()` | Click-based connection highlighting |

---

## 9. Database & Scheduler

### 9.1 SQLite Schema

Three tables:

**optimization_jobs:**
```sql
CREATE TABLE optimization_jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT,
    created_at TEXT,
    completed_at TEXT,
    parameters TEXT  -- JSON
);
```

**solutions:**
```sql
CREATE TABLE solutions (
    solution_id TEXT PRIMARY KEY,
    job_id TEXT,
    created_at TEXT,
    business_impact TEXT,   -- JSON
    disruption_metrics TEXT -- JSON
);
```

**dealer_changes:**
```sql
CREATE TABLE dealer_changes (
    change_id INTEGER PRIMARY KEY AUTOINCREMENT,
    solution_id TEXT,
    dealer_id TEXT,
    from_ftc_id TEXT,
    to_ftc_id TEXT,
    impact_score REAL
);
```

### 9.2 Scheduler

The `OptimizationScheduler` wraps APScheduler for daily re-optimization:

| Property | Default |
|----------|---------|
| Schedule | Daily at 2:00 AM |
| Max concurrent jobs | 1 |
| Config source | `parameters.json` → `scheduler.*` |

Provides `trigger_now()` for on-demand runs via the API.

---

## 10. Configuration Reference

### 10.1 `config/parameters.json`

```json
{
    "data_paths": {
        "dealers": "data/dealers.parquet",
        "ftcs": "data/ftcs.parquet",
        "relationships": "data/relationships.parquet",
        "proximity": "data/proximity.parquet"
    },
    "optimization": {
        "alpha_1": 0.5,
        "alpha_2": 0.3,
        "lambda": 0.1,
        "time_limit": 3600,
        "optimality_gap": 0.01
    },
    "clustering": {
        "method": "kmeans",
        "min_cluster_size": 3,
        "max_cluster_size": 25,
        "cluster_ratio": 1.5
    },
    "solver": {
        "time_limit_seconds": 300,
        "num_workers": 4,
        "optimality_gap": 0.05
    },
    "scheduler": {
        "enabled": true,
        "cron_hour": 2,
        "cron_minute": 0,
        "max_concurrent_jobs": 1
    },
    "constraints": {
        "max_dealers_per_ftc": 25,
        "min_dealers_per_ftc": 1,
        "max_travel_radius_km": 50,
        "workload_balance_threshold": 0.3
    },
    "validation": {
        "dealer_type_values": ["static", "mobile"],
        "latitude_range": [-90, 90],
        "longitude_range": [-180, 180],
        "ntb_share_range": [0, 1]
    }
}
```

### 10.2 All Override Keys

Passed via CLI `--time-limit` or API `parameters` object:

| Dot-notation Key | Type | Section |
|-----------------|------|---------|
| `optimization.alpha_1` | float | optimization |
| `optimization.alpha_2` | float | optimization |
| `optimization.lambda` | float | optimization |
| `solver.time_limit_seconds` | int | solver |
| `solver.num_workers` | int | solver |
| `solver.optimality_gap` | float | solver |
| `constraints.max_dealers_per_ftc` | int | constraints |
| `constraints.min_dealers_per_ftc` | int | constraints |
| `constraints.max_travel_radius_km` | float | constraints |
| `constraints.workload_balance_threshold` | float | constraints |

---

## 11. CLI Reference

### 11.1 Generate Synthetic Data

```bash
python main.py generate --dealers 5000 --ftcs 500 --output data --seed 42
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dealers` | 5000 | Number of dealers |
| `--ftcs` | 500 | Number of FTCs |
| `--output` | `data` | Output directory for parquet files |
| `--seed` | 42 | Random seed for reproducibility |

### 11.2 Run Optimization

```bash
python main.py optimize --time-limit 60 --report
```

| Flag | Description |
|------|-------------|
| `--time-limit` | Solver time limit in seconds |
| `--report` | Generate text report file |

### 11.3 Start Server

```bash
python main.py serve --host 0.0.0.0 --port 5000 --debug
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | 5000 | Port number |
| `--debug` | False | Enable Flask debug mode |

---

## 12. Complete Change Log

### 12.1 Solver Rewrite — Nearest-Feasible-FTC + Repair Phases

Replaced the original greedy+simulated-annealing solver with a deterministic nearest-feasible-FTC assignment followed by 5 repair/refinement phases:

- **Phase 1**: Initial cluster-to-FTC assignment (6 trials, most-constrained-first ordering)
- **Phase 2**: Overloaded FTC repair (move clusters to next-nearest feasible FTC)
- **Phase 3**: Unused FTC activation (move nearest eligible cluster)
- **Phase 4**: Swap refinement (pairwise cluster swaps to reduce distance)
- **Phase 5**: Post-swap overload repair + final unused FTC activation

**Files:** `optimization/solver.py`

### 12.2 Workload Capacity Bug Fix

The normalized workload (~0.5) vs normalized capacity (~0.5) comparison made it impossible for any FTC to hold more than ~1 dealer. Removed the workload capacity check entirely — only `max_dealers` constraint is enforced.

**Files:** `optimization/solver.py`

### 12.3 Undefined Variable Fixes

Fixed `util_penalty` undefined bug and `best_iter` undefined bug that caused pipeline crashes during repair phases.

**Files:** `optimization/solver.py`

### 12.4 Distance Metrics Added

Added `mean_distance_km` and `median_distance_km` to solver result, pipeline result, and report output.

**Files:** `optimization/solver.py`, `pipeline.py`, `analysis/reporting.py`

### 12.5 Synthetic Data Improvements

- FTCs anchored on random dealer locations within each city (~100–800 m jitter)
- Jitter uses uniform random angle + radius (not Gaussian) to ensure FTCs never sit directly on top of a dealer
- Increased default clusters via `min_cluster_size: 3` and `cluster_ratio: 1.5`

**Files:** `data/generate_synthetic_data.py`, `config/parameters.json`

### 12.6 Feasibility Mask Distance Bug Fix

The feasibility mask was using normalized [0,1] distance values instead of raw kilometer values, making the `max_travel_radius_km` constraint non-functional. Fixed by storing a copy of the raw km matrix and using it for feasibility computation.

**Files:** `features/distance.py`, `pipeline.py`, `optimization/model.py`

### 12.7 Feasibility Mask Cluster Expansion

Added cluster cohesion expansion: within each cluster, unionize feasibility masks so all dealers share the same feasible FTC set. This prevents the solver from getting stuck when the intersection of feasible FTCs across cluster members is empty (common with tight 50 km radius).

**Files:** `optimization/model.py`

### 12.8 Post-Solve Feasibility Enforcement

Added two-pass enforcement after solving that moves any dealer assigned to an FTC beyond their individual (pre-expansion) feasibility mask to the nearest truly feasible FTC. All metrics (changed dealers, mean distance, etc.) are recomputed after enforcement.

**Files:** `optimization/model.py`

### 12.9 Disruption Minimization Toggle

Added UI checkbox "Minimize Disruption" that passes `lambda = 5.0` to the solver. Phase 1 prefers keeping clusters at their current FTC when feasible. Phase 4 swap refinement is skipped. Orange dealer markers indicate unchanged dealers.

**Files:** `static/index.html`, `static/app.js`, `optimization/solver.py`

### 12.10 Distance Limit Toggle

Added UI checkbox + number input for max dealer-FTC distance. When checked, passes `constraints.max_travel_radius_km` value. When unchecked, passes `999` (unlimited).

**Files:** `static/index.html`, `static/app.js`

### 12.11 UI Distance Visualization

- `renderDistances()` shows FTC→dealer km labels at midpoints (text-shadow only, no box)
- Dealer↔dealer distances shown as polygon edges (capped at 10 dealers)
- Toggled via "Show Distances (km)" checkbox
- Haversine formula in JavaScript matches Python implementation

**Files:** `static/app.js`, `static/style.css`

### 12.12 Measurement Tool

- Crosshair cursor when active
- Click to place numbered waypoints
- Lines between consecutive points with segment labels
- Running total in floating panel
- Toggled via "📏 Measure Distance" checkbox

**Files:** `static/app.js`, `static/style.css`, `static/index.html`

### 12.13 Optimization Stats Panel

Collapsible sidebar panel showing:
- Dealers retained/changed (%) — computed from `data.changes`
- Unallocated dealers — dealers with no FTC assignment
- Active/idle FTCs — utilization stats
- Min/Max/Mean/Median dealers per active FTC — distribution stats
- Color-coded values (green = good, amber = warning, red = bad)

**Files:** `static/index.html`, `static/app.js`, `static/style.css`

### 12.14 Relationship Sampling Fix

The `/data/dealers` endpoint sampled to 1000 dealers while `/data/relationships` returned all relationships, causing mismatches when users increased dealer counts. Fixed by filtering relationships to match the sampled dealer/FTC ID sets. Later removed all sampling to honor user-input counts.

**Files:** `api/routes.py`

### 12.15 FTC-Distance Line Colors

Lines from FTC to unchanged dealers rendered in green (`#22c55e`) instead of translucent white. Dealer markers stay orange for unchanged.

**Files:** `static/app.js`

### 12.16 Synthetic Data FTC Jitter Fix

Replaced Gaussian jitter (`np.random.normal(0, 0.005)`) with uniform random angle + radius in [0.001°, 0.008°] (~100–800 m), ensuring FTCs never land directly on top of a dealer.

**Files:** `data/generate_synthetic_data.py`

### 12.17 Colab Self-Contained Notebook

Created `territory_optimizer_colab.ipynb` — a 237 KB notebook with all 44 source files embedded via `%%writefile`, plus install, data generation, ngrok tunnel, and server startup cells.

**Files:** `territory_optimizer_colab.ipynb`

### 12.18 Documentation

Created `docs/architecture.md` — this file.

---

## File Map

```
territory_optimizer/
├── __init__.py
├── main.py                          # CLI entry point
├── pipeline.py                      # 7-step optimization pipeline
├── scheduler.py                     # APScheduler wrapper
├── requirements.txt                 # Python dependencies
├── pytest.ini                       # Pytest config
├── territory_optimizer_colab.ipynb  # Self-contained Colab notebook
│
├── config/
│   ├── __init__.py
│   ├── parameters.json              # All default parameters
│   └── settings.py                  # ConfigManager (dot-notation access)
│
├── data/
│   ├── __init__.py
│   ├── generate_synthetic_data.py   # Synthetic data generator
│   ├── loader.py                    # DataLoader (parquet/CSV I/O)
│   ├── processor.py                 # DataProcessor (index maps, assignment matrix)
│   └── validator.py                 # DataValidator (integrity checks)
│
├── features/
│   ├── __init__.py
│   ├── workload.py                  # WorkloadEngineer
│   ├── capacity.py                  # CapacityEngineer
│   ├── compatibility.py             # CompatibilityEngineer
│   └── distance.py                  # DistanceEngineer (Haversine)
│
├── optimization/
│   ├── __init__.py
│   ├── clustering.py                # SpatialClusterer (constrained K-Means)
│   ├── constraints.py               # OptimizationConstraints (feasibility, validation)
│   ├── model.py                     # TerritoryModel (orchestrator + post-solve enforcement)
│   └── solver.py                    # TerritorySolver (Greedy + CP-SAT)
│
├── api/
│   ├── __init__.py
│   ├── routes.py                    # Flask REST endpoints
│   ├── schemas.py                   # Pydantic models
│   └── server.py                    # Flask app factory
│
├── database/
│   ├── __init__.py
│   ├── connection.py                # SQLite connection manager
│   └── models.py                    # Dataclasses + CRUD models
│
├── analysis/
│   ├── __init__.py
│   ├── impact.py                    # (placeholder)
│   ├── reporting.py                 # ReportGenerator (text reports)
│   ├── results.py                   # ResultAnalyzer (business impact)
│   └── scheduler.py                 # (placeholder)
│
├── static/
│   ├── index.html                   # Dashboard HTML
│   ├── app.js                       # Dashboard JavaScript (Leaflet.js)
│   └── style.css                    # Dashboard styles
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py                 # Data generation + validation tests
│   ├── test_features.py             # Workload/Capacity/Compatibility tests
│   ├── test_model.py                # Integration test (model + constraints)
│   └── test_optimization.py         # Clustering + solver tests
│
└── docs/
    └── architecture.md              # This file
```

---

*Generated: June 2026*
