from typing import Dict, Optional

_WEIGHTS = {
    "sleep_duration": 0.20,
    "step_count": 0.15,
    "exercise_duration": 0.15,
    "water_intake": 0.10,
    "heart_rate": 0.15,
    "bmi": 0.15,
    "stress_level": 0.10,
}

def _score_in_range(value: float, low: float, high: float) -> float:
    """1.0 inside [low, high], linearly decaying to 0 as it moves further away."""
    if low <= value <= high:
        return 1.0
    span = max(high - low, 1e-6)
    distance = (low - value) if value < low else (value - high)
    return max(0.0, 1.0 - distance / span)


def _score_at_least(value: float, target: float) -> float:
    if value >= target:
        return 1.0
    return max(0.0, value / target)


def compute_health_score(inputs: Dict) -> Optional[int]:
    component_scores = {}
    if inputs.get("sleep_duration") is not None:
        component_scores["sleep_duration"] = _score_in_range(inputs["sleep_duration"], 7, 9)
    if inputs.get("step_count") is not None:
        component_scores["step_count"] = _score_at_least(inputs["step_count"], 8000)
    if inputs.get("exercise_duration") is not None:
        component_scores["exercise_duration"] = _score_at_least(inputs["exercise_duration"], 30)
    if inputs.get("water_intake") is not None:
        component_scores["water_intake"] = _score_at_least(inputs["water_intake"], 2.5)
    if inputs.get("heart_rate") is not None:
        component_scores["heart_rate"] = _score_in_range(inputs["heart_rate"], 60, 80)
    if inputs.get("bmi") is not None:
        component_scores["bmi"] = _score_in_range(inputs["bmi"], 18.5, 24.9)
    if inputs.get("stress_level") is not None:
        stress = str(inputs["stress_level"]).lower()
        component_scores["stress_level"] = 0.3 if stress == "high" else (0.7 if stress == "medium" else 1.0)

    if not component_scores:
        return None

    total_weight = sum(_WEIGHTS[k] for k in component_scores)
    weighted_sum = sum(_WEIGHTS[k] * v for k, v in component_scores.items())

    return int(round(100 * weighted_sum / total_weight))
