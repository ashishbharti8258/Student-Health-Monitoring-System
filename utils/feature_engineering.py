import pandas as pd

ENGINEERED_FEATURES = [
    "sedentary_bmi_strain",
    "stress_sleep_deficit",
    "toxic_cardio_load",
    "cardiovascular_load",
    "metabolic_intensity",
    "hydration_ratio",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["sedentary_bmi_strain"] = df["bmi"] * df.get("physical_activity_level_sedentary", 0)
    df["stress_sleep_deficit"] = df["sleep_duration"] / (df.get("stress_level_high", 0) + 1)
    df["toxic_cardio_load"] = df["heart_rate"] * df.get("smoking_alcohol_yes", 0)
    df["cardiovascular_load"] = df["heart_rate"] * df["bmi"]
    df["metabolic_intensity"] = df["calorie_expenditure"] / (df["step_count"] + 1)
    df["hydration_ratio"] = df["water_intake"] / (df["calorie_expenditure"] + 1)

    return df
