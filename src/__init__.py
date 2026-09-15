"""
Analysis runners for the ABM Engine.

Each module in this package owns the command-line arguments and
orchestration logic for exactly one analysis mode (robustness testing,
Sobol sensitivity analysis, SHAP-based surrogate explanation). main.py
dispatches to these modules; it does not duplicate their argument
definitions or execution logic.
"""