import streamlit as st

from utils.data_loader import get_active_dataset, render_data_uploader, using_uploaded_data, active_source_label
from utils.health_score import compute_health_score
from utils.recommendations import generate_recommendations
from utils import visualization as viz

st.set_page_config(page_title="Student Analysis", layout="wide")
st.title("🧑‍🎓 Individual Student Analysis")

render_data_uploader(key_prefix="student")
df = get_active_dataset()

if using_uploaded_data():
    st.caption(f"📊 Looking up students in {active_source_label()} — {len(df):,} rows.")

if df is None:
    st.warning(
        "⚠️ No dataset found at `data/student_health_data.csv`, and nothing uploaded yet. "
        "Upload a CSV in the sidebar, or drop one in the `data/` folder, to look up real student records."
    )
    st.stop()

id_col = next((c for c in ["student_id", "id"] if c in df.columns), None)

if id_col:
    student_id = st.text_input(f"Enter {id_col.replace('_', ' ').title()}:", value=str(df[id_col].iloc[0]))
    matches = df[df[id_col].astype(str) == str(student_id)]
    row = matches.iloc[0] if not matches.empty else None
    if row is None:
        st.error(f"No student found with {id_col} = {student_id}")
        st.stop()
else:
    row_idx = st.number_input("Enter row number:", min_value=0, max_value=len(df) - 1, value=0, step=1)
    row = df.iloc[int(row_idx)]

st.divider()
c1, c2 = st.columns([1, 2])

with c1:
    with st.container(border=True):
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
        st.write(f"**ID:** {row.get(id_col, row.name) if id_col else row.name}")
        if "gender" in row:
            st.write(f"**Gender:** {row['gender']}")
        if "age" in row:
            st.write(f"**Age:** {row['age']}")
        if "physical_activity_level" in row:
            st.write(f"**Activity Level:** {row['physical_activity_level']}")

with c2:
    st.subheader("Biometric Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Heart Rate", f"{row['heart_rate']:.0f} bpm" if "heart_rate" in row else "N/A")
    m2.metric("Sleep Duration", f"{row['sleep_duration']:.1f}h" if "sleep_duration" in row else "N/A")
    m3.metric("Stress Level", str(row.get("stress_level", "N/A")))

    inputs = {
        "sleep_duration": row.get("sleep_duration"),
        "step_count": row.get("step_count"),
        "exercise_duration": row.get("exercise_duration"),
        "water_intake": row.get("water_intake"),
        "heart_rate": row.get("heart_rate"),
        "bmi": row.get("bmi"),
        "stress_level": row.get("stress_level"),
    }
    score = compute_health_score(inputs)

    st.subheader("Health Score Profile")
    if score is not None:
        st.plotly_chart(viz.gauge_chart(score, title="Composite Health Score"), use_container_width=True)
    else:
        st.info("Not enough data in this row to compute a health score.")

st.divider()
st.subheader("💡 Personalized Recommendations")
for tip in generate_recommendations(inputs):
    st.write(f"- {tip}")