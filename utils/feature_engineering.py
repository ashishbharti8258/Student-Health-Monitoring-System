import pandas as pd

# List of new engineered features added
ENGINEERED_FEATURES = [
    "sedentary_bmi_strain",
    "stress_sleep_deficit",
    "toxic_cardio_load",
    "cardiovascular_load",
    "metabolic_intensity",
    "hydration_ratio",
]


def add_engineered_features(df):
    df = df.copy()

    # 1. Sedentary BMI strain
    if "physical_activity_level_sedentary" in df.columns:
        df["sedentary_bmi_strain"] = df["bmi"] * df["physical_activity_level_sedentary"]
    else:
        df["sedentary_bmi_strain"] = 0

    # 2. Stress sleep deficit
    if "stress_level_high" in df.columns:
        df["stress_sleep_deficit"] = df["sleep_duration"] / (df["stress_level_high"] + 1)
    else:
        df["stress_sleep_deficit"] = df["sleep_duration"] / 1.0

    # 3. Toxic cardio load (heart rate * smoking/alcohol)
    if "smoking_alcohol_yes" in df.columns:
        df["toxic_cardio_load"] = df["heart_rate"] * df["smoking_alcohol_yes"]
    else:
        df["toxic_cardio_load"] = 0

    # 4. Cardio load
    df["cardiovascular_load"] = df["heart_rate"] * df["bmi"]

    # 5. Metabolic intensity (calories burned per step)
    df["metabolic_intensity"] = df["calorie_expenditure"] / (df["step_count"] + 1)

    # 6. Hydration ratio
    df["hydration_ratio"] = df["water_intake"] / (df["calorie_expenditure"] + 1)

    return df