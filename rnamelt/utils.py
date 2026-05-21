"""
utils.py — shared helpers for RNA melting analysis.
"""

import numpy as np

#R = 8.314  # J/(mol·K)


def celsius_to_kelvin(T_celsius):
    return np.array(T_celsius, dtype=float) + 273.15


def kelvin_to_celsius(T_kelvin):
    return np.array(T_kelvin, dtype=float) - 273.15


def safe_json(obj):
    """Recursively convert numpy scalars to Python-native types for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return [safe_json(x) for x in obj.tolist()]
    if isinstance(obj, Exception):
        return str(obj)
    return obj
