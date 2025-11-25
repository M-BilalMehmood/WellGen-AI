#!/usr/bin/env python3
"""Quick launcher for WellGen AI - automatically loads .env"""

import sys
import os
from pathlib import Path

# Add parent directory to path to find .env file
parent_dir = Path(__file__).parent.parent
if (parent_dir / '.env').exists():
    print(f"✓ Found .env file in {parent_dir}")
else:
    print(f"⚠️  No .env file found in {parent_dir}")
    print("Looking in current directory...")
    if Path('.env').exists():
        print("✓ Found .env in current directory")

# Now import and run the main application
try:
    from src.wellgen_rag import WellGenRAG
    
    print("\n" + "="*70)
    print("WellGen AI - Personalized Wellness Coach with RAG")
    print("Powered by Kaggle nutrition data + Groq AI")
    print("="*70)
    
    # Initialize system
    wellgen = WellGenRAG(use_rag=True)
    
    print("\n" + "="*70)
    print("Let's create your personalized wellness profile!")
    print("="*70)
    
    # Collect user profile
    age = int(input("\nYour age: "))
    gender = input("Your gender (male/female/other): ").lower()
    height = float(input("Your height in cm: "))
    weight = float(input("Your current weight in kg: "))
    
    print("\nGoals:")
    print("1. Weight Loss")
    print("2. Muscle Gain")
    print("3. Maintain Weight")
    print("4. General Health")
    goal_map = {'1': 'weight_loss', '2': 'muscle_gain', '3': 'maintain', '4': 'health'}
    goal_input = input("Choose your goal (1-4): ")
    goal = goal_map.get(goal_input, 'weight_loss')
    
    allergies = input("Any food allergies? (or 'none'): ")
    cuisine = input("Preferred cuisine? (e.g., Italian, Asian, Mediterranean, or 'any'): ") or "any"
    
    # Calculate BMI
    bmi, bmi_category = wellgen.calculate_bmi(height, weight)
    
    # Create user_data dictionary for the system
    user_data = {
        'age': age,
        'gender': gender,
        'height': height,
        'weight': weight,
        'bmi': bmi,
        'bmi_category': bmi_category,
        'goal': goal,
        'allergies': allergies.lower(),
        'cuisine': cuisine.lower()
    }
    
    # Set profile
    wellgen.user_profile = user_data
    
    print("\n" + "="*70)
    print("Your Profile:")
    print("="*70)
    print(f"- Age: {age}")
    print(f"- Gender: {gender.title()}")
    print(f"- Height: {height} cm")
    print(f"- Weight: {weight} kg")
    print(f"- BMI: {bmi} ({bmi_category})")
    print(f"- Goal: {goal.replace('_', ' ').title()}")
    print(f"- Allergies: {allergies}")
    print(f"- Cuisine: {cuisine.title()}")
    
    # Generate diet plan
    print("\n" + "="*70)
    print("Generating your personalized 7-day diet plan...")
    print("(Using RAG to retrieve relevant nutrition info from Kaggle datasets)")
    print("="*70)
    
    diet_plan = wellgen.generate_diet_plan(user_data)
    print("\n" + diet_plan)
    
    # Interactive chat
    print("\n" + "="*70)
    print("Ask me anything about nutrition, diet, or wellness!")
    print("Type 'quit' or 'exit' to end the conversation")
    print("="*70)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\n👋 Thanks for using WellGen AI! Stay healthy!")
            break
        
        # Check if user wants a NEW diet plan
        create_keywords = ['create', 'generate', 'new', 'make me', 'give me']
        plan_keywords = ['diet plan', 'meal plan', 'food plan']
        
        wants_new_plan = any(create in user_input.lower() for create in create_keywords) and \
                         any(plan in user_input.lower() for plan in plan_keywords)
        
        if wants_new_plan:
            confirm = input("\nCreate a new diet plan? (yes/no): ").lower()
            if confirm in ['yes', 'y']:
                print("\n📋 Let me create a personalized diet plan for you!\n")
                try:
                    height = int(input("Height (cm): ") or str(int(user_data['height'])))
                    weight = int(input("Weight (kg): ") or str(int(user_data['weight'])))
                    age = int(input("Age: ") or str(user_data['age']))
                    gender = input(f"Gender ({user_data['gender']}): ").lower() or user_data['gender']
                    allergies = input(f"Allergies ({user_data['allergies']}): ") or user_data['allergies']
                    cuisine = input(f"Cuisine ({user_data['cuisine']}): ") or user_data['cuisine']
                    
                    print("\nGoal options:")
                    print("1. Weight Loss")
                    print("2. Muscle Gain")
                    print("3. Maintain Weight")
                    print("4. General Health")
                    goal_choice = input("Choose goal (1-4): ").strip()
                    goal = goal_map.get(goal_choice, user_data['goal'])
                    
                    # Update user data
                    user_data.update({
                        'height': height,
                        'weight': weight,
                        'age': age,
                        'gender': gender,
                        'allergies': allergies,
                        'cuisine': cuisine,
                        'goal': goal
                    })
                    
                    # Recalculate BMI
                    bmi, bmi_category = wellgen.calculate_bmi(height, weight)
                    user_data['bmi'] = bmi
                    user_data['bmi_category'] = bmi_category
                    wellgen.user_profile = user_data
                    
                    print("\n🤖 Generating your evidence-based diet plan...\n")
                    plan = wellgen.generate_diet_plan(user_data)
                    print("="*70)
                    print(plan)
                    print("="*70)
                    continue
                    
                except ValueError:
                    print("\n❌ Invalid input. Please enter valid numbers.")
                    continue
        
        # Let the chat function handle all questions naturally with RAG context
        response = wellgen.chat(user_input)
        print(f"\nAssistant: {response}")

except KeyboardInterrupt:
    print("\n\n👋 Goodbye! Stay healthy!")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure GROQ_API_KEY is set in your .env file")
    print("2. Check that all required packages are installed:")
    print("   pip install python-dotenv groq sentence-transformers faiss-cpu")
    sys.exit(1)
