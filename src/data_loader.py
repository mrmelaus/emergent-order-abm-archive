"""
Data Loading Utilities for the ABM Engine.

This module centralizes all static input-data loading routines shared
across every analysis mode (robustness testing, Sobol sensitivity
analysis, SHAP-based surrogate explanation, and parameter ablation
studies).

Loading logic is intentionally decoupled from any particular analysis
mode: every runner receives the same immutable inputs (voyage
groupings, historical drought series, and the colony-stress fallback
map) and applies its own sampling or intervention strategy on top of
them. No random sampling occurs in this module.
"""

import os

import pandas as pd

from src.worker import load_and_group_by_voyage
from src.config import build_year_stress_map


def load_simulation_inputs(convicts_path, drought_path):
    """
    Load and prepare all static inputs required by the ABM engine.

    Parameters
    ----------
    convicts_path : str
        Path to the raw convict records CSV file.

    drought_path : str
        Path to the historical annual drought index CSV file.

    Returns
    -------
    dict
        Dictionary with the following keys:

        grouped_voyages : dict
            Mapping of voyage_id -> DataFrame of convict records.

        voyage_ids : list
            List of available voyage identifiers.

        drought_history_source : list of float
            Annual drought index values, ordered chronologically.

        year_stress_map : dict
            Mapping of transported_year -> mean historical colony stress.

    Raises
    ------
    FileNotFoundError
        If either input file does not exist.

    ValueError
        If the drought file does not contain the expected column.
    """
    if not os.path.exists(convicts_path):
        raise FileNotFoundError(f"Convict records file not found: {convicts_path}")

    if not os.path.exists(drought_path):
        raise FileNotFoundError(f"Drought history file not found: {drought_path}")

    print("Loading convict data...")
    grouped_voyages, voyage_ids = load_and_group_by_voyage(convicts_path)
    print(f"Loaded {len(voyage_ids):,} voyages.")

    print("Loading historical drought data...")
    climate_df = pd.read_csv(drought_path)

    if "Drought_Index" not in climate_df.columns:
        raise ValueError("Drought data must contain a 'Drought_Index' column.")

    drought_history_source = climate_df["Drought_Index"].astype(float).tolist()
    print(f"Loaded {len(drought_history_source):,} annual drought observations.")

    print("Building historical year-stress map...")
    year_stress_map = build_year_stress_map(convicts_path)
    print(f"Loaded {len(year_stress_map):,} years with stress data.")

    return {
        "grouped_voyages": grouped_voyages,
        "voyage_ids": voyage_ids,
        "drought_history_source": drought_history_source,
        "year_stress_map": year_stress_map,
    }


def default_data_paths(project_root):
    """
    Resolve the default input file locations relative to the project root.

    Parameters
    ----------
    project_root : str
        Absolute path to the project root directory.

    Returns
    -------
    tuple of (str, str)
        (convicts_path, drought_path)
    """
    convicts_path = os.path.join(project_root, "data", "raw", "raw_convicts_with_dest.csv")
    drought_path = os.path.join(project_root, "data", "processed", "quarterly_drought.csv")
    return convicts_path, drought_path