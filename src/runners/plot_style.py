"""
Shared matplotlib house style for all figure-generating runners
(sobol_runner.py, shap_runner.py, robustness_runner.py, etc.).
"""

import matplotlib.pyplot as plt

# ===============================================================
# COLOR PALETTE - (print-friendly, colorblind-safe)
# ===============================================================

COLORS = {
    # Primary palette (colorblind-safe)
    "blue": "#4C72B0",      # Primary - main bars, lines (dark blue)
    "orange": "#DD8452",    # Secondary - comparison, second category (burnt orange)
    "red": "#C44E52",       # Highlight - anomalies, alerts (crimson)
    
    # Extended palette (for additional categories)
    "green": "#55A868",     # Use only when NOT paired with red!
    "purple": "#8172B2",    # Tertiary - additional category
    "teal": "#4C8C8A",      # Additional - extra category
    
    # Neutral palette
    "black": "#333333",     # Text and axes
    "grey": "#999999",      # Background elements
    "light_grey": "#E6E6E6", # Grid lines, subtle elements
    
    # Colormap for continuous data (heatmaps, gradients)
    "colormap": "Blues",    # ColorBrewer sequential blue
    "colormap_diverging": "RdBu",  # Red-Blue diverging
}

# ===============================================================
# Standard figure sizes (inches)
# ===============================================================

FIGSIZE_STANDARD = (8, 5)      # single-panel bar/line charts
FIGSIZE_WIDE = (10, 5)         # dual-panel or wide comparison charts
FIGSIZE_SQUARE = (6, 6)        # heatmaps / interaction matrices
FIGSIZE_TALL = (8, 7)          # dense multi-series convergence plots
heatmap_figsize = FIGSIZE_SQUARE

def apply_house_style():
    """Call once, before creating any figure in a runner script."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "Source Sans Pro",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.labelweight": "normal",
    })

def save_figure(fig, path, tight=True):
    """Standard save call — keeps DPI and bbox handling consistent."""
    if tight:
        fig.savefig(path, bbox_inches="tight", dpi=plt.rcParams["savefig.dpi"])
    else:
        fig.savefig(path, dpi=plt.rcParams["savefig.dpi"])