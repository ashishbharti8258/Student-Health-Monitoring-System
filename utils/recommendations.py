from typing import Dict, List


def generate_recommendations(inputs: Dict) -> List[str]:
    tips = []

    sleep_duration = inputs.get("sleep_duration")
    if sleep_duration is not None and sleep_duration < 6:
        tips.append("Consider improving sleep duration and maintaining a consistent sleep schedule. "
                     "Most students benefit from 7-9 hours of sleep per night.")

    physical_activity_level = str(inputs.get("physical_activity_level", "")).lower()
    exercise_duration = inputs.get("exercise_duration")
    if physical_activity_level == "sedentary" or (exercise_duration is not None and exercise_duration < 15):
        tips.append("Consider gradually increasing regular physical activity, "
                     "even short daily walks can help.")

    step_count = inputs.get("step_count")
    if step_count is not None and step_count < 5000:
        tips.append("Your daily step count is on the lower side — try building in more movement "
                     "throughout the day.")

    water_intake = inputs.get("water_intake")
    if water_intake is not None and water_intake < 2:
        tips.append("Consider maintaining adequate hydration throughout the day.")

    stress_level = str(inputs.get("stress_level", "")).lower()
    if stress_level == "high":
        tips.append("Consider stress management techniques (breathing exercises, breaks, journaling) "
                     "and seeking appropriate support when needed.")

    sleep_quality = str(inputs.get("sleep_quality", "")).lower()
    if sleep_quality in ("poor", "low"):
        tips.append("Poor sleep quality can compound stress and fatigue — a consistent wind-down "
                     "routine before bed may help.")

    smoking_alcohol = str(inputs.get("smoking_alcohol", "")).lower()
    if smoking_alcohol == "yes":
        tips.append("Reducing smoking/alcohol use is one of the highest-impact changes for "
                     "long-term cardiovascular health.")

    bmi = inputs.get("bmi")
    if bmi is not None and (bmi < 18.5 or bmi >= 25):
        tips.append("Your BMI falls outside the typical healthy range — consider discussing "
                     "nutrition and activity habits with a healthcare professional.")

    if not tips:
        tips.append("Your inputs look broadly consistent with healthy habits — keep it up!")

    tips.append("These are general wellness suggestions only, not a medical diagnosis or treatment plan.")

    return tips