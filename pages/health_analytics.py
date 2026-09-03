import streamlit as st
import numpy as np  # BUG FIX: original file used np.random.randn(...) below but never imported numpy

from utils.data_loader import get_active_dataset, render_data_uploader, using_uploaded_data, active_source_label, numeric_columns
from utils import visualization as viz

st.set_page_config(page_title="Health Analytics", layout="wide")
st.title("🔬 Health Analytics")

render_data_uploader(key_prefix="analytics")
df = get_active_dataset()

if using_uploaded_data():
    st.caption(f"📊 Analyzing {active_source_label()} — {len(df):,} rows.")

if df is None:
    st.warning(
        "⚠️ No dataset found at `data/student_health_data.csv`, and nothing uploaded yet. "
        "Showing a random demo scatter — upload a CSV in the sidebar for real analytics."
    )
    import pandas as pd
    col_x, col_y = "sleep_duration", "heart_rate"
    demo_df = pd.DataFrame(np.random.randn(100, 2), columns=[col_x, col_y])
    st.subheader(f"Relationship: {col_x} vs {col_y} (demo)")
    st.scatter_chart(demo_df)
    st.stop()

num_cols = numeric_columns(df)
if len(num_cols) < 2:
    st.info("The dataset needs at least two numeric columns for correlation analysis.")
    st.stop()

with st.expander("🔍 Correlation Controls", expanded=True):
    col_x = st.selectbox("Select X Axis", num_cols, index=0)
    col_y = st.selectbox("Select Y Axis", num_cols, index=min(1, len(num_cols) - 1))

st.subheader(f"Relationship: {col_x} vs {col_y}")
color_col = "health_condition" if "health_condition" in df.columns else None
st.plotly_chart(viz.scatter(df, col_x, col_y, color=color_col), use_container_width=True)

corr_value = df[col_x].corr(df[col_y])
with st.container(border=True):
    direction = "positive" if corr_value >= 0 else "inverse"
    strength = "strong" if abs(corr_value) >= 0.5 else ("moderate" if abs(corr_value) >= 0.2 else "weak")
    st.info(
        f"💡 **Statistical Note:** {col_x.replace('_',' ').title()} and {col_y.replace('_',' ').title()} "
        f"show a {strength} {direction} correlation (r = {corr_value:.2f})."
    )

st.divider()
st.subheader("Correlation Heatmap")
default_cols = num_cols[:8]
selected_cols = st.multiselect("Columns to include", num_cols, default=default_cols)
if len(selected_cols) >= 2:
    st.plotly_chart(viz.correlation_heatmap(df, selected_cols), use_container_width=True)
else:
    st.info("Select at least two columns to render a correlation heatmap.")