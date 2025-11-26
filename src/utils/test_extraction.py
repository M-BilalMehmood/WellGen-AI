"""
Test script for body part extraction from diet plan.
This shows what the fixed code should extract and generate.
"""

# Sample diet plan text from user
diet_plan_text = """
**Body parts most affected by this diet:**

    Belly: Reduced fat storage and improved insulin sensitivity
    Legs: Improved muscle tone and strength
    Arms: Improved muscle tone and strength
    Shoulders: Improved muscle tone and strength
    Chest: Improved muscle tone and strength
    Back: Improved muscle tone and strength
"""

# Mapping body parts to exercises (from your LoRA training data)
def map_body_part_to_exercise(body_part):
    exercise_map = {
        "belly": "plank",
        "abdomen": "plank",
        "legs": "squat",
        "arms": "barbell biceps curl",
        "shoulders": "shoulder press",
        "chest": "bench press",
        "back": "deadlift",
    }
    
    for keyword, exercise in exercise_map.items():
        if keyword in body_part.lower():
            return exercise
    return "plank"

# Extract and map
import re
pattern = r"\*\*Body [Pp]arts [Mm]ost [Aa]ffected.*?:\*\*\s*\n(.*?)(?=\n\*\*|\Z)"
match = re.search(pattern, diet_plan_text, re.DOTALL | re.IGNORECASE)

if match:
    section = match.group(1)
    parts = re.findall(r"^\s*(.+?):", section, re.MULTILINE)
    
    print("Extracted body parts:")
    for part in parts:
        print(f"  - {part.strip()}")
    
    print("\nMapped to exercises:")
    exercises = []
    for part in parts:
        exercise = map_body_part_to_exercise(part.strip())
        exercises.append(exercise)
        print(f"  - {part.strip()} → {exercise}")
    
    # Remove duplicates
    unique_exercises = list(dict.fromkeys(exercises))[:3]
    
    print("\nFinal exercises for image generation:")
    for ex in unique_exercises:
        prompt = f"{ex}, muscle anatomy visualization"
        print(f"  Prompt: '{prompt}'")
