import pandas as pd
import pytest


@pytest.fixture
def raw_rows() -> pd.DataFrame:
    """Three rows shaped like the source CSV: pure, water, melamine."""
    return pd.DataFrame(
        {
            "Sample_ID": ["S1", "S2", "S3"],
            "Collection_Date": ["2022-01-01", "2022-01-02", "2022-01-03"],
            "Collection_Point": ["Farm Gate", "Processing Plant", "Retail Outlet"],
            "State": ["Karnataka", "Kerala", "Goa"],
            "Milk_Type": ["Toned", "Toned", "Skimmed"],
            "Fat_percent": [3.5, 2.1, 0.4],
            "SNF_percent": [8.9, 7.2, 9.0],
            "Specific_Gravity": [1.030, 1.020, 1.031],
            "pH": [6.7, 6.9, 6.6],
            "Freezing_Point_C": [-0.53, -0.45, -0.53],
            "Conductivity_mS_cm": [4.8, 4.2, 4.9],
            "Is_Adulterated": [0, 1, 1],
            "Adulterant_Detected": [None, "Water", "Melamine"],
            "Unused_Lab_Column": [1, 2, 3],
        }
    )
