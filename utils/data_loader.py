"""
Shared, cached dataset loading across three possible sources:

  1. Live PostgreSQL/Supabase data      (default, whenever it's configured)
  2. A CSV the user uploaded this session
  3. The bundled sample CSV (data/student_health_data.csv)

Every analysis/prediction page calls render_data_uploader() once (renders
the sidebar source picker + upload widget) and then get_active_dataset()
to fetch whichever source is currently selected. Because the database is
now the default whenever it's configured, pages query Supabase live
instead of a local CSV without the user having to do anything extra.
"""

from typing import Optional

import pandas as pd
import streamlit as st

from utils.config import DATA_PATH
from utils.validator import check_dataset_compatibility
from utils import db as db_module

SESSION_KEY = "uploaded_dataset"
SESSION_NAME_KEY = "uploaded_dataset_name"
SOURCE_KEY = "data_source"  # "database" | "upload" | "sample"

SOURCE_LABELS = {
    "database": "🗄️ Live Database (Supabase)",
    "upload": "📤 My Uploaded File",
    "sample": "📁 Sample Dataset",
}


@st.cache_data(show_spinner=False)
def load_dataset() -> Optional[pd.DataFrame]:
    """Load the bundled sample dataset, or return None if it isn't present."""
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include="number").columns.tolist()


def _available_sources() -> list:
    sources = []
    if db_module.is_configured():
        sources.append("database")
    if SESSION_KEY in st.session_state:
        sources.append("upload")
    sources.append("sample")
    return sources


def render_data_uploader(key_prefix: str = "") -> None:
    """
    Render the sidebar: a data-source picker (Live DB / Upload / Sample)
    plus the CSV upload widget with validation and an optional
    "Save to Cloud Database" action.
    """
    with st.sidebar:
        st.markdown("#### 🗂️ Data Source")

        options = _available_sources()
        current = st.session_state.get(SOURCE_KEY)
        if current not in options:
            # Default preference: live database > uploaded file > sample CSV.
            current = options[0]
            st.session_state[SOURCE_KEY] = current

        choice = st.radio(
            "Analyze data from:",
            options,
            format_func=lambda o: SOURCE_LABELS[o],
            index=options.index(current),
            key=f"{key_prefix}_source_radio",
        )
        st.session_state[SOURCE_KEY] = choice

        if choice == "database":
            st.caption("Querying Supabase/PostgreSQL live — no local CSV involved.")

        st.divider()
        st.markdown("#### 📤 Upload a CSV")
        uploaded = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            key=f"{key_prefix}_uploader",
            help=(
                "Needs these columns (case/space-insensitive): sleep_duration, "
                "heart_rate, bmi, calorie_expenditure, step_count, "
                "exercise_duration, water_intake."
            ),
        )

        if uploaded is not None:
            try:
                new_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Couldn't read that file: {e}")
            else:
                new_df.columns = [c.strip().lower().replace(" ", "_") for c in new_df.columns]
                ok, missing = check_dataset_compatibility(new_df)
                if not ok:
                    st.error("Missing required column(s): " + ", ".join(missing))
                elif st.session_state.get(SESSION_NAME_KEY) != uploaded.name:
                    st.session_state[SESSION_KEY] = new_df
                    st.session_state[SESSION_NAME_KEY] = uploaded.name
                    st.session_state[SOURCE_KEY] = "upload"
                    st.success(f"Loaded **{uploaded.name}** — {len(new_df):,} rows.")
                    st.rerun()  # refresh so "My Uploaded File" appears as a source option

        if SESSION_KEY in st.session_state:
            st.caption(f"Uploaded file ready: **{st.session_state.get(SESSION_NAME_KEY, 'upload')}**")

            if db_module.is_configured():
                save_mode = st.radio(
                    "Save mode",
                    ["Append to database", "Replace database contents"],
                    key=f"{key_prefix}_save_mode",
                    horizontal=True,
                )
                if st.button("☁️ Save to Cloud Database", key=f"{key_prefix}_save_db", use_container_width=True):
                    mode = "replace" if save_mode.startswith("Replace") else "append"
                    try:
                        with st.spinner("Saving to Supabase/PostgreSQL..."):
                            n_written = db_module.save_dataframe_to_db(st.session_state[SESSION_KEY], mode=mode)
                    except Exception as e:
                        st.error(f"Save failed: {e}")
                    else:
                        st.success(f"Saved {n_written:,} rows to the cloud database.")
                        st.session_state[SOURCE_KEY] = "database"
                        st.rerun()
            else:
                st.caption("💡 Configure PostgreSQL in secrets.toml to save uploads permanently to the cloud.")

            if st.button("🗑️ Remove uploaded file", key=f"{key_prefix}_clear", use_container_width=True):
                st.session_state.pop(SESSION_KEY, None)
                st.session_state.pop(SESSION_NAME_KEY, None)
                if st.session_state.get(SOURCE_KEY) == "upload":
                    st.session_state.pop(SOURCE_KEY, None)
                st.rerun()


def get_active_dataset() -> Optional[pd.DataFrame]:
    """
    Return whichever dataset is currently selected. Live database is the
    default whenever Postgres/Supabase is configured, so pages read real,
    current rows straight out of the cloud instead of a local CSV.
    """
    source = st.session_state.get(SOURCE_KEY, "sample")

    if source == "database":
        live_df = db_module.fetch_dataframe()
        if live_df is not None and not live_df.empty:
            return live_df
        # Configured but empty/unreachable right now - fall back gracefully.

    if source == "upload" and SESSION_KEY in st.session_state:
        return st.session_state[SESSION_KEY]

    return load_dataset()


def using_uploaded_data() -> bool:
    return st.session_state.get(SOURCE_KEY) in ("upload", "database")


def active_source_label() -> str:
    source = st.session_state.get(SOURCE_KEY, "sample")
    if source == "database":
        return "live Supabase/PostgreSQL data"
    if source == "upload" and SESSION_KEY in st.session_state:
        return f"your uploaded file ({st.session_state.get(SESSION_NAME_KEY, 'upload')})"
    return "the sample dataset"