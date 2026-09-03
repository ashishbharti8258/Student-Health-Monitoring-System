"""
Shared, cached dataset loading.

This module is new - the original pages never actually loaded
data/student_health_data.csv at all; every page used np.random / hardcoded
placeholder values instead. Every page now goes through load_dataset() so
they all see the same (cached) DataFrame with normalized column names.
"""

from typing import Optional

import pandas as pd
import streamlit as st

from utils.config import DATA_PATH


@st.cache_data(show_spinner=False)
def load_dataset() -> Optional[pd.DataFrame]:
    """Load the student health dataset, or return None if it isn't present yet."""
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include="number").columns.tolist()