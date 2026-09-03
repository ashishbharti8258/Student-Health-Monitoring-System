import pandas as pd

# Required health features in standard lowercase
REQUIRED_COLUMNS = [
    'sleep_duration',
    'heart_rate',
    'bmi',
    'calorie_expenditure',
    'step_count',
    'exercise_duration',
    'water_intake'
]


def check_dataset_compatibility(df: pd.DataFrame):
    """
    Checks if the uploaded DataFrame contains the mandatory health features,
    ignoring case, spaces, and underscores.
    """
    df_cols_normalized = [
        col.strip().lower().replace(" ", "_") for col in df.columns
    ]

    missing_cols = []
    for req_col in REQUIRED_COLUMNS:
        req_normalized = req_col.strip().lower().replace(" ", "_")
        if req_normalized not in df_cols_normalized:
            missing_cols.append(req_col)

    if len(missing_cols) > 0:
        return False, missing_cols

    return True, []