# Update Image Generation for Diet App

## Location
File: `app.py`
Function: `show_generating()`
Lines: ~581-593

## Current Code (REMOVE THIS):
```python
                for part in affected_parts:
                    # Map anatomical part to exercise name
                    exercise = map_body_part_to_exercise(part)
                    
                    # Use simple prompt matching LoRA training format:
                    # "exercise_name, muscle anatomy visualization"
                    prompt = f"{exercise}, muscle anatomy visualization"
                    
                    print(f"🖼️  Generating: '{prompt}' for {part}")
                    
                    image_path = generate_ai_image(prompt, "generated_body_parts")
                    if image_path:
                        images.append((exercise.title(), image_path))
```

## New Code (REPLACE WITH THIS):
```python
                for part in affected_parts:
                    gender = st.session_state.user_profile.get('gender', 'male')
                    
                    # Anatomical body part visualization for diet app
                    prompt = f"anatomical illustration of {gender} {part}, fitness body diagram, clean medical style, highlighted muscle area"
                    
                    print(f"🖼️  Generating: {part}")
                    
                    image_path = generate_ai_image(prompt, "generated_body_parts")
                    if image_path:
                        images.append((part.title(), image_path))
```

## What Changed:
1. **Removed** exercise mapping (`map_body_part_to_exercise`)
2. **Changed** prompt from exercise-based to body part anatomical visualization
3. **Added** gender to make visualization more personalized
4. **Shows** actual body parts (belly, chest, legs) instead of exercises (plank, bench press, squat)

This is better for a DIET app - showing which body areas will be affected, not gym exercises!
