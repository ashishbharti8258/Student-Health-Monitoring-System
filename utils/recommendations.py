from typing import Dict, List

def generate_recommendations(inputs):
    # getting tips 
    tips = []
    
    # check sleep
    sleep_duration = inputs.get("sleep_duration")
    if sleep_duration and sleep_duration < 6:
        tips.append("Try to sleep more! Most students need at least 7-9 hours.")

    # check physical activity and exercise
    activity = inputs.get("physical_activity_level", "").lower()
    exercise = inputs.get("exercise_duration")
    if activity == "sedentary" or (exercise is not None and exercise < 15):
        tips.append("You should try to move more or go for short daily walks.")

    # steps check
    steps = inputs.get("step_count")
    if steps and steps < 5000:
        tips.append("Your step count is a bit low, try walking around more.")

    # water intake
    water = inputs.get("water_intake")
    if water and water < 2:
        tips.append("Drink more water throughout the day to stay hydrated.")

    # stress
    stress = inputs.get("stress_level", "").lower()
    if stress == "high":
        tips.append("You're stressed! Try breathing exercises, take breaks, or talk to someone.")

    # sleep quality
    sleep_q = inputs.get("sleep_quality", "").lower()
    if sleep_q in ["poor", "low"]:
        tips.append("Poor sleep quality makes fatigue worse — try having a wind-down routine.")

    # smoking/alcohol
    smoke = inputs.get("smoking_alcohol", "").lower()
    if smoke == "yes":
        tips.append("Try cutting down on smoking/alcohol for better long-term health.")

    # bmi check
    bmi = inputs.get("bmi")
    if bmi and (bmi < 18.5 or bmi >= 25):
        tips.append("Your BMI is outside the normal range, maybe consult a health professional.")

    # fallback if everything is fine
    if not tips:
        tips.append("Looking good! Your habits seem healthy, keep it up.")

    # standard disclaimer
    tips.append("Disclaimer: Just general wellness tips, not a real medical diagnosis.")
    
    return tips