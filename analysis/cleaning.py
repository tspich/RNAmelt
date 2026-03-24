"""
cleaning.py — parse and validate RNA melting CSV data.

Expected format:
  Column 0 : Temperature (°C)
  Column 1+: Absorbance or fluorescence signal (one per sample)
"""

import pandas as pd
#import numpy as np


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows that are entirely NaN
    df = df.dropna(how="all")

    # Coerce all columns to numeric (non-parseable → NaN)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where temperature is missing
    df = df.dropna(subset=[df.columns[0]])

    # Note: Should be the case anyway
    ## Sort by temperature ascending
    #df = df.sort_values(df.columns[0]).reset_index(drop=True)

    # Rename columns for clarity
    cols = [df.columns[0]] + list(df.columns[1:])
    rename = {cols[0]: "temperature"}
    for i, c in enumerate(cols[1:], 1):
        rename[c] = f"signal_{i}" if str(c).startswith("Unnamed") or str(c).strip() == "" else str(c)
    df = df.rename(columns=rename)

    return df


def get_signal_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c != "temperature"]
