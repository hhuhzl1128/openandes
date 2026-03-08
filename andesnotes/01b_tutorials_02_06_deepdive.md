# ANDES Study Notes — Tutorials 02–06 Deep Dive

Sources:
- 02 First Simulation
- 03 Power Flow
- 04 Time-Domain Simulation
- 05 Data and File Formats
- 06 Plotting Results

## 02 First Simulation
Minimal loop:
1. Load case
2. Run power flow
3. Configure TDS horizon
4. Run dynamic simulation
5. Plot key state variables (for example generator speed)

## 03 Power Flow
- Establish operating point before dynamic studies
- Confirm convergence and inspect mismatch trajectory
- Treat non-convergence as a model/data quality issue

## 04 Time-Domain Simulation
- Use validated power-flow state as initialization
- Set simulation horizon and disturbance timing carefully
- Inspect physical consistency of trajectories, not just successful completion

## 05 Data and File Formats
- Understand case formats and output artifacts
- Keep input/output organization explicit for reproducibility

## 06 Plotting Results
- Build repeatable plotting scripts
- Use overlay plots for scenario comparison
- Keep figure naming and run IDs consistent
