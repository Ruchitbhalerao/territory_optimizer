# Territory Optimizer

Enterprise territory optimization system for dealer-to-FTC assignment rebalancing.

## Overview
This system provides a data-driven approach for assigning territories and micro-locations. It groups ~100K dealers into optimal micro-location clusters using constrained K-Means, and optimally assigns ~30K field employees (FTCs) using a Google OR-Tools CP-SAT solver, balancing workloads, minimizing travel distance, and enforcing capacity limits while minimizing disruptions.

## Project Structure
- `data/` - Synthetic data generation and data pipeline (loading, validation, processing)
- `features/` - Feature engineering (capacity, compatibility, distance, workload)
- `optimization/` - Spatial clustering, OR-Tools solver, constraints, and MILP formulation
- `api/` - Flask REST API for running optimizations and fetching solutions
- `database/` - SQLite storage for jobs, solutions, and dealer reassignments
- `analysis/` - Business impact calculation and reporting
- `config/` - System configuration (parameters, scheduler, solver)

## Setup & Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate synthetic data (creates 5000 dealers and 500 FTCs by default):
   ```bash
   python main.py generate --dealers 5000 --ftcs 500
   ```

## Usage

Run a one-shot optimization from the CLI:
```bash
python main.py optimize --report
```
*This runs the pipeline, solves the assignment, and generates `optimization_report.txt`.*

Start the API server (also starts the periodic re-optimization scheduler):
```bash
python main.py serve --port 5000
```

## API Endpoints
- `GET /api/v1/health` - Check health status
- `POST /api/v1/optimize` - Trigger optimization run
- `GET /api/v1/solution/<job_id>` - Get solution details
- `GET /api/v1/solution/<job_id>/changes` - Get dealer reassignment list
- `GET /api/v1/scheduler/status` - Get scheduler status

## Scheduler
The periodic re-optimizer runs daily at 2:00 AM by default, governed by `config/parameters.json`.
