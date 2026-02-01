"""
Data Loading Module
===================

Functions for loading and parsing Bloomberg financial data.
Handles the specific date encoding issue in the Bloomberg export format.
"""

import numpy as np
import pandas as pd
from typing import Optional


def excel_serial_to_datetime(serial: pd.Series) -> pd.Series:
    """
    Convert Excel serial dates to pandas datetime.
    """
    return pd.to_datetime("1899-12-30") + pd.to_timedelta(serial, unit="D")


def load_bloomberg_csv(csv_path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load Bloomberg data from CSV and return a wide-format DataFrame.
    
    Handles the Bloomberg export date encoding issue where dates may be 
    stored as "1970-01-01 00:00:00.000XXXXX" where XXXXX is the Excel serial.
    """
    df = pd.read_csv(csv_path)
    df.columns = ["date", "ticker", "price"]

    # Clean ticker names and convert prices
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    s = df["date"].astype(str)

    # Handle buggy dates: "1970-01-01 00:00:00.000XXXXX" -> XXXXX = Excel serial
    excel_pattern = r"^1970-01-01\s+00:00:00\.0*(\d+)$"
    serial_str = s.str.extract(excel_pattern, expand=False)
    serial = pd.to_numeric(serial_str, errors="coerce")
    excel_dates = excel_serial_to_datetime(serial)

    # Normal dates (2004-10-25, etc.)
    normal_dates = pd.to_datetime(s, format="mixed", errors="coerce")

    # Combine: use Excel dates where detected, otherwise normal dates
    is_excel_date = serial.notna()
    df["date"] = normal_dates.where(~is_excel_date, excel_dates)

    # Clean and pivot
    df = df.dropna(subset=["date", "ticker"])
    df = df.sort_values(["date", "ticker"])
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")

    wide = df.pivot(index="date", columns="ticker", values="price").sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide = wide.replace([np.inf, -np.inf], np.nan)
    wide = wide.dropna(axis=1, how="all")

    if verbose:
        print(f"Loaded data: {wide.index.min().date()} -> {wide.index.max().date()}")
        print(f"Shape: {wide.shape[0]} days × {wide.shape[1]} tickers")

    return wide
