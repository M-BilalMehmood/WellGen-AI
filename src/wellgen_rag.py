#!/usr/bin/env python3
"""WellGen AI with RAG (Retrieval Augmented Generation) - Production Version."""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from groq import Groq
import os
from pathlib import Path
from dotenv import load_dotenv
from .rag_system import RAGSystem

# Load environment variables from .env file
load_dotenv()

class WellGenRAG:
    def __init__(self, use_rag=True):
        print("Loading WellGen AI with RAG...")
        
        self.use_rag = use_rag
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Groq API
        api_key = os.getenv('GROQ_API_KEY')
        if api_key:
            self.groq_client = Groq(api_key=api_key)
            print("✓ Groq API configured")
        else:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        # Initialize RAG system
        if self.use_rag:
            self.rag = RAGSystem()
        
        # Conversation memory
        self.conversation_history = []
        self.user_profile = None
        self.current_plan = None
    
    def calculate_bmi(self, height_cm, weight_kg):
        """Calculate BMI and category."""
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 25:
            category = "Normal weight"
        elif 25 <= bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"
        
        return round(bmi, 1), category
    
    def calculate_calories(self, weight_kg, height_cm, age, gender, goal):
        """Calculate daily calorie needs."""
        if gender.lower() == 'male':
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
        
        tdee = bmr * 1.55
        
        if goal == "weight_loss":
            return int(tdee - 500)
        elif goal == "muscle_gain":
            return int(tdee + 300)
        else:
            return int(tdee)
    
    def validate_diet_plan(self, diet_plan, user_data):
        """Cross-validate the generated diet plan using Groq AI."""
        print("\n🔍 Validating diet plan for quality and safety...")
        
        target_calories = self.calculate_calories(
            user_data['weight'], 
            user_data['height'], 
            user_data['age'], 
            user_data['gender'], 
            user_data['goal']
        )
        
        validation_prompt = f"""You are a nutrition expert validator. Review this diet plan critically and check for:

USER PROFILE:
- Age: {user_data['age']}, Gender: {user_data['gender']}
- Height: {user_data['height']}cm, Weight: {user_data['weight']}kg
- Goal: {user_data['goal'].replace('_', ' ')}
- Allergies: {user_data['allergies']}
- Cuisine: {user_data.get('cuisine', 'any')}
- Target calories: {target_calories} cal/day

DIET PLAN TO VALIDATE:
{diet_plan}

VALIDATION CHECKLIST:
1. Are all 7 days present with distinct meals? (Monday-Sunday)
2. Does each day have breakfast, lunch, dinner, and 2 snacks?
3. Are daily calories close to target ({target_calories} ± 200 cal)?
4. Are allergenic foods ({user_data['allergies']}) completely avoided?
5. Is the plan nutritionally balanced (protein, carbs, fats, fiber)?
6. Are there any extreme, unsafe, or scientifically unsound recommendations?
7. Are portion sizes realistic and clearly specified?
8. Is there sufficient VARIETY? (No repetitive meals across days)
9. Does it respect the cuisine preference ({user_data.get('cuisine', 'any')})?

Respond in this exact format:
VALIDATION RESULT: [PASS/WARNING/FAIL]
ISSUES FOUND: [list each issue on a new line, or "None"]
SUGGESTIONS: [list improvements, or "None"]
OVERALL ASSESSMENT: [1-2 sentence summary]"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a strict nutrition expert validator. Be thorough and critical in your assessment."},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.3,  # Lower temperature for consistent validation
                max_tokens=1024
            )
            
            validation_result = response.choices[0].message.content
            
            # Parse validation result
            result_lower = validation_result.lower()
            if "validation result: pass" in result_lower:
                print("✓ Validation passed!")
                if "issues found: none" not in result_lower:
                    print("\n📝 Validator notes:")
                    print(validation_result.split("ISSUES FOUND:")[1].split("SUGGESTIONS:")[0].strip())
                return True, validation_result
            elif "validation result: warning" in result_lower:
                print("⚠️  Validation passed with warnings:")
                print(validation_result.split("ISSUES FOUND:")[1].split("SUGGESTIONS:")[0].strip())
                return True, validation_result
            else:
                print("❌ Validation failed! Issues detected:")
                print(validation_result.split("ISSUES FOUND:")[1].split("SUGGESTIONS:")[0].strip())
                return False, validation_result
                
        except Exception as e:
            print(f"⚠️  Validation error: {e}")
            print("Proceeding with plan (validation unavailable)")
            return True, "Validation error - proceeding"
    
    def generate_diet_plan(self, user_data):
        """Generate personalized diet plan using RAG + Groq."""
        height = user_data.get('height', 175)
        weight = user_data.get('weight', 84)
        age = user_data.get('age', 30)
        gender = user_data.get('gender', 'male')
        goal = user_data.get('goal', 'weight_loss')
        allergies = user_data.get('allergies', 'none')
        cuisine = user_data.get('cuisine', 'any')
        
        self.user_profile = user_data
        
        bmi, bmi_category = self.calculate_bmi(height, weight)
        calories = self.calculate_calories(weight, height, age, gender, goal)
        
        # Create base query for RAG retrieval
        goal_text = goal.replace('_', ' ')
        allergy_text = f" avoiding {allergies}" if allergies != 'none' else ""
        cuisine_text = f" {cuisine} cuisine" if cuisine != 'any' else ""
        rag_query = f"personalized diet plan for {goal_text}{allergy_text}{cuisine_text}"
        
        # Get relevant nutrition knowledge via RAG
        augmented_prompt, retrieved_docs = self.rag.augment_prompt(rag_query, user_data)
        
        # Create comprehensive prompt with retrieved knowledge
        prompt = f"""{augmented_prompt}

Generate a detailed, personalized 7-day diet plan based on the above knowledge and user profile.

REQUIREMENTS:
1. Create 7 DIFFERENT days of meals (Monday-Sunday)
2. Each day should have:
   - Breakfast (with calories)
   - Lunch (with calories)
   - Dinner (with calories)
   - 2 Snacks (with calories)
3. **CRITICAL: MEAL VARIETY IS MANDATORY**
   - Do NOT repeat the same breakfast every day.
   - Do NOT repeat the same lunch or dinner.
   - Provide diverse options (e.g., different proteins, grains, vegetables).
4. **CUISINE PREFERENCE**: {cuisine}
   - Incorporate {cuisine} flavors and dishes where appropriate.
   - If 'any', provide a mix of cuisines.
5. Consider the user's allergies: {allergies}
6. Include specific portion sizes
7. Total daily calories should match target: {calories} cal
8. Use the nutrition knowledge provided above to ensure scientifically sound recommendations

FORMAT:
Day: [Day Name]
Breakfast: [specific meal with portions] ([calories] cal)
Lunch: [specific meal with portions] ([calories] cal)
Dinner: [specific meal with portions] ([calories] cal)
Snacks: [Snack 1] ([cal] cal), [Snack 2] ([cal] cal)

Also provide:
- Expected results in 8 weeks
- Body parts most affected by this diet
- 5 key success tips based on nutrition science
- Important warnings
- Next steps

Be specific, professional, and evidence-based."""

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are WellGen AI, a professional wellness coach and nutritionist. Use evidence-based nutrition science to provide personalized advice."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4096
                )
                generated_plan = response.choices[0].message.content
                
                # Show which knowledge was used
                print("\n📚 Used nutrition knowledge:")
                for doc in retrieved_docs:
                    print(f"  - {doc['title']}")
                print()
                
                # Validate the diet plan
                is_valid, validation_feedback = self.validate_diet_plan(generated_plan, user_data)
                
                if is_valid:
                    self.current_plan = generated_plan
                    return generated_plan
                else:
                    if attempt < max_retries - 1:
                        print(f"\n🔄 Regenerating diet plan (attempt {attempt + 2}/{max_retries})...")
                        # Add validation feedback to prompt for retry
                        prompt += f"\n\nPREVIOUS ATTEMPT HAD ISSUES:\n{validation_feedback}\n\nPlease address these issues in your new plan."
                    else:
                        print("\n⚠️  Returning plan despite validation warnings. Please review carefully.")
                        self.current_plan = generated_plan
                        return generated_plan
                        
            except Exception as e:
                print(f"⚠️  Groq API error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                else:
                    return "Error generating diet plan. Please try again."
    
    def chat(self, question):
        """Chat using RAG + Groq with conversation history and context awareness."""
        # Build context about the user's profile and current plan
        context_parts = []
        
        if self.user_profile:
            profile_summary = f"""USER PROFILE:
- Age: {self.user_profile.get('age')}
- Gender: {self.user_profile.get('gender')}
- Height: {self.user_profile.get('height')}cm
- Weight: {self.user_profile.get('weight')}kg
- BMI: {self.user_profile.get('bmi')} ({self.user_profile.get('bmi_category')})
- Goal: {self.user_profile.get('goal', '').replace('_', ' ')}
- Allergies: {self.user_profile.get('allergies')}"""
            context_parts.append(profile_summary)
        
        if self.current_plan:
            context_parts.append(f"\nGENERATED DIET PLAN:\n{self.current_plan}")
        
        # Use RAG to retrieve relevant context
        if self.use_rag:
            augmented_prompt, retrieved_docs = self.rag.augment_prompt(question, self.user_profile)
        else:
            augmented_prompt = question
            retrieved_docs = []
        
        # Build the full prompt with context
        if context_parts:
            full_context = "\n\n".join(context_parts)
            final_prompt = f"{full_context}\n\nUSER QUESTION: {augmented_prompt}"
        else:
            final_prompt = augmented_prompt
        
        try:
            # Build messages with conversation history
            messages = [
                {"role": "system", "content": """You are WellGen AI, a professional wellness coach and nutritionist. 

IMPORTANT INSTRUCTIONS:
1. You have access to the user's profile and their personalized diet plan in the context above.
2. When users ask about "the diet plan" or "the plan I was given", refer to the GENERATED DIET PLAN in the context.
3. Maintain conversation continuity - remember what was discussed before.
4. Provide specific, actionable advice based on their profile and plan.
5. Be supportive, encouraging, and empathetic - especially when users express concerns or fears.
6. Use evidence-based nutrition science in your responses."""}
            ]
            
            # Add conversation history (keep last 10 messages for context window)
            for msg in self.conversation_history[-10:]:
                messages.append(msg)
            
            # Add current question
            messages.append({"role": "user", "content": final_prompt})
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
            
            assistant_response = response.choices[0].message.content
            
            # Store in conversation history
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            if retrieved_docs and len(retrieved_docs) > 0:
                print(f"\n📚 (Referenced: {retrieved_docs[0]['title']})")
            
            return assistant_response
        except Exception as e:
            print(f"⚠️  Groq API error: {e}")
            return "Sorry, I encountered an error. Please try again."

def main():
    wellgen = WellGenRAG(use_rag=True)
    
    print("\n" + "="*70)
    print("WellGen AI - RAG-Powered Wellness Coach")
    print("="*70)
    print("🚀 Powered by: Groq (Llama 3.3 70B) + RAG System")
    print("📚 Knowledge Base: 20+ nutrition science documents")
    print("🔍 Retrieval: Semantic search with sentence transformers")
    
    print("\nI can help you with:")
    print("  1. Generate personalized diet plans")
    print("  2. Answer health and nutrition questions")
    print("  3. Provide evidence-based wellness advice")
    print("\nType 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nStay healthy! 👋")
            break
        
        if not user_input:
            continue
        
        # Check if user wants a NEW diet plan
        create_keywords = ['create', 'generate', 'new', 'make me', 'give me']
        plan_keywords = ['diet plan', 'meal plan', 'food plan']
        
        wants_new_plan = any(create in user_input.lower() for create in create_keywords) and \
                         any(plan in user_input.lower() for plan in plan_keywords)
        
        if wants_new_plan and wellgen.current_plan is None:
            needs_diet_plan = True
        elif wants_new_plan and wellgen.current_plan is not None:
            confirm = input("\nYou already have a diet plan. Create a new one? (yes/no): ").lower()
            needs_diet_plan = confirm in ['yes', 'y']
        else:
            needs_diet_plan = False
        
        if needs_diet_plan:
            print("\n📋 Let me create a personalized diet plan for you!\n")
            
            try:
                height = int(input("Height (cm): ") or "175")
                weight = int(input("Weight (kg): ") or "84")
                age = int(input("Age: ") or "30")
                gender = input("Gender (male/female): ").lower() or "male"
                allergies = input("Any allergies? (or 'none'): ") or "none"
                
                print("\nGoal options:")
                print("  1. Weight loss")
                print("  2. Muscle gain")
                print("  3. Diabetes management")
                goal_choice = input("Choose goal (1-3): ").strip()
                
                goal_map = {"1": "weight_loss", "2": "muscle_gain", "3": "diabetes"}
                goal = goal_map.get(goal_choice, "weight_loss")
                
                user_data = {
                    'height': height,
                    'weight': weight,
                    'age': age,
                    'gender': gender,
                    'goal': goal,
                    'allergies': allergies
                }
                
                print("\n🤖 Generating your evidence-based diet plan...\n")
                
                plan = wellgen.generate_diet_plan(user_data)
                
                print("="*70)
                print(plan)
                print("="*70)
                
            except ValueError:
                print("\n❌ Invalid input. Please enter valid numbers.")
                continue
        else:
            # General chat with RAG
            print("\nWellGen AI: ", end="", flush=True)
            response = wellgen.chat(user_input)
            print(response + "\n")

if __name__ == "__main__":
    main()
