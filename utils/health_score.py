from typing import Dict, Optional
# Weights for calculating overall health score
WEIGHTS = {
    "sleep_duration": 0.20,
    "step_count": 0.15,
    "exercise_duration": 0.15,
    "water_intake": 0.10,
    "heart_rate": 0.15,
    "bmi": 0.15,
    "stress_level": 0.10,
}


def compute_health_score(inputs):
    scores = {}

    # 1. Sleep score (ideal 7-9 hrs)
    sleep = inputs.get("sleep_duration")
    if sleep is not None:
        if 7 <= sleep <= 9:
            scores["sleep_duration"] = 1.0
        elif sleep < 7:
            scores["sleep_duration"] = max(0.0, sleep / 7.0)
        else:
            scores["sleep_duration"] = max(0.0, 1.0 - (sleep - 9) / 2.0)

    # 2. Steps score (target 8000)
    steps = inputs.get("step_count")
    if steps is not None:
        scores["step_count"] = min(1.0, steps / 8000.0)

    # 3. Exercise score (target 30 mins)
    exercise = inputs.get("exercise_duration")
    if exercise is not None:
        scores["exercise_duration"] = min(1.0, exercise / 30.0)

    # 4. Water intake score (target 2.5L)
    water = inputs.get("water_intake")
    if water is not None:
        scores["water_intake"] = min(1.0, water / 2.5)

    # 5. Heart rate score (ideal 60-80 bpm)
    hr = inputs.get("heart_rate")
    if hr is not None:
        if 60 <= hr <= 80:
            scores["heart_rate"] = 1.0
        elif hr < 60:
            scores["heart_rate"] = max(0.0, hr / 60.0)
        else:
            scores["heart_rate"] = max(0.0, 1.0 - (hr - 80) / 20.0)

    # 6. BMI score (ideal 18.5 - 24.9)
    bmi = inputs.get("bmi")
    if bmi is not None:
        if 18.5 <= bmi <= 24.9:
            scores["bmi"] = 1.0
        elif bmi < 18.5:
            scores["bmi"] = max(0.0, bmi / 18.5)
        else:
            scores["bmi"] = max(0.0, 1.0 - (bmi - 24.9) / 6.4)

    # 7. Stress level score
    stress = inputs.get("stress_level")
    if stress is not None:
        stress_str = str(stress).lower()
        if stress_str == "high":
            scores["stress_level"] = 0.3
        elif stress_str == "medium":
            scores["stress_level"] = 0.7
        else:
            scores["stress_level"] = 1.0

    if not scores:
        return None

    # Calculate final weighted average score out of 100
    total_w = sum(WEIGHTS[k] for k in scores)
    weighted_sum = sum(WEIGHTS[k] * v for k, v in scores.items())

    final_score = round((weighted_sum / total_w) * 100)
    return int(final_score)