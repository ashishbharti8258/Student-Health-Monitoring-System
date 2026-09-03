"""
Cloud PostgreSQL backend for real-time data ingestion + dynamic metric
tracking (e.g. Neon, Supabase, RDS, Render Postgres, etc. all work the
same way since this just uses a standard connection string).

Configure credentials in `.streamlit/secrets.toml` (see
`.streamlit/secrets.toml.example` in this project):

    [postgres]
    host = "your-db-host.example.com"
    port = 5432
    dbname = "health_monitoring"
    user = "your_user"
    password = "your_password"
    sslmode = "require"

Environment variables (POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DBNAME,
POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_SSLMODE) are used as a fallback,
so this also works when deployed somewhere other than Streamlit Cloud.

If no credentials are found anywhere, every function here degrades
gracefully (returns None / False) instead of crashing the app - the rest
of the dashboard keeps working off the sample CSV / uploaded file.
"""

import os
from typing import Dict, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

TABLE_NAME = "health_records"

REQUIRED_COLUMNS = [
    "sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
    "step_count", "exercise_duration", "water_intake",
]


def _get_conf(key: str, default=None):
    try:
        return st.secrets["postgres"][key]
    except Exception:
        return os.environ.get(f"POSTGRES_{key.upper()}", default)


def is_configured() -> bool:
    return bool(_get_conf("host")) and bool(_get_conf("dbname")) and bool(_get_conf("user"))


@st.cache_resource(show_spinner=False)
def get_engine() -> Optional[Engine]:
    """One pooled connection shared across the whole app/session."""
    if not is_configured():
        return None
    host = _get_conf("host")
    port = _get_conf("port", 5432)
    dbname = _get_conf("dbname")
    user = _get_conf("user")
    password = _get_conf("password", "")
    sslmode = _get_conf("sslmode", "require")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?sslmode={sslmode}"
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def init_schema() -> None:
    """Create the health_records table if it doesn't exist yet (idempotent)."""
    engine = get_engine()
    if engine is None:
        return
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        student_id TEXT,
        gender TEXT,
        age INTEGER,
        sleep_duration DOUBLE PRECISION,
        heart_rate DOUBLE PRECISION,
        bmi DOUBLE PRECISION,
        calorie_expenditure DOUBLE PRECISION,
        step_count DOUBLE PRECISION,
        exercise_duration DOUBLE PRECISION,
        water_intake DOUBLE PRECISION,
        stress_level TEXT,
        physical_activity_level TEXT,
        health_condition TEXT,
        recorded_at TIMESTAMPTZ DEFAULT now()
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def insert_record(record: Dict) -> None:
    """Real-time ingestion: insert one new health record."""
    engine = get_engine()
    if engine is None:
        raise RuntimeError(
            "PostgreSQL isn't configured. Add credentials to "
            "`.streamlit/secrets.toml` (see secrets.toml.example)."
        )
    init_schema()
    clean = {k: v for k, v in record.items() if v is not None}
    columns = list(clean.keys())
    col_list = ", ".join(columns)
    placeholders = ", ".join(f":{c}" for c in columns)
    stmt = text(f"INSERT INTO {TABLE_NAME} ({col_list}) VALUES ({placeholders})")
    with engine.begin() as conn:
        conn.execute(stmt, clean)
    # Invalidate cached reads so the new row/metrics show up immediately.
    fetch_dataframe.clear()
    fetch_live_metrics.clear()


@st.cache_data(ttl=15, show_spinner=False)
def fetch_dataframe(limit: int = 5000) -> Optional[pd.DataFrame]:
    """Most recent rows from Postgres. Short TTL keeps this feeling live
    without hitting the database on every single widget interaction."""
    engine = get_engine()
    if engine is None:
        return None
    init_schema()
    query = text(f"SELECT * FROM {TABLE_NAME} ORDER BY recorded_at DESC LIMIT :limit")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"limit": limit})
    return df


@st.cache_data(ttl=15, show_spinner=False)
def fetch_live_metrics() -> Optional[Dict]:
    """Dynamic metric tracking: KPI aggregates computed in SQL (fast even
    as the table grows large), refreshed every ~15 seconds."""
    engine = get_engine()
    if engine is None:
        return None
    init_schema()
    query = text(f"""
        SELECT
            COUNT(*)                AS total_records,
            AVG(sleep_duration)     AS avg_sleep,
            AVG(bmi)                AS avg_bmi,
            AVG(heart_rate)         AS avg_heart_rate,
            AVG(step_count)         AS avg_steps,
            MAX(recorded_at)        AS last_ingested_at
        FROM {TABLE_NAME}
    """)
    with engine.connect() as conn:
        row = conn.execute(query).mappings().first()
    return dict(row) if row else None


# Columns the health_records table actually has (mirrors the DDL in
# init_schema). Anything outside this list in an uploaded file is dropped;
# anything missing is written as NULL.
TABLE_COLUMNS = [
    "student_id", "gender", "age",
    "sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
    "step_count", "exercise_duration", "water_intake",
    "stress_level", "physical_activity_level", "health_condition",
]


def save_dataframe_to_db(df: pd.DataFrame, mode: str = "append") -> int:
    """
    Persist an uploaded DataFrame into the cloud Postgres table, so it's no
    longer just a one-session upload but part of the durable dataset.

    mode:
        "append"  - add these rows to whatever's already in the table.
        "replace" - wipe the table first, then insert only these rows.

    Returns the number of rows written. Column names are normalized the
    same way as everywhere else in the app (lowercase, spaces -> underscores).
    """
    engine = get_engine()
    if engine is None:
        raise RuntimeError(
            "PostgreSQL isn't configured. Add credentials to "
            "`.streamlit/secrets.toml` (see secrets.toml.example)."
        )
    init_schema()

    clean = df.copy()
    clean.columns = [c.strip().lower().replace(" ", "_") for c in clean.columns]
    for col in TABLE_COLUMNS:
        if col not in clean.columns:
            clean[col] = None
    clean = clean[TABLE_COLUMNS]

    if mode == "replace":
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {TABLE_NAME} RESTART IDENTITY"))

    clean.to_sql(TABLE_NAME, engine, if_exists="append", index=False, method="multi", chunksize=500)

    # Invalidate cached reads so the newly saved rows show up immediately.
    fetch_dataframe.clear()
    fetch_live_metrics.clear()
    return len(clean)