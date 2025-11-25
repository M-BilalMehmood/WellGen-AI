#!/usr/bin/env python3
"""
Convert Kaggle nutrition datasets to RAG knowledge base format.
Creates comprehensive JSON knowledge base from real nutrition data.
"""

import pandas as pd
import json
import os
from typing import List, Dict

def create_food_nutrition_docs(base_path: str) -> List[Dict]:
    """Convert FINAL FOOD DATASET to knowledge base documents"""
    docs = []
    doc_id = 1000
    
    # Load all 5 food groups - sample 100 from each for efficiency
    for group_num in range(1, 6):
        filepath = os.path.join(base_path, f"FINAL FOOD DATASET/FOOD-DATA-GROUP{group_num}.csv")
        if not os.path.exists(filepath):
            continue
            
        df = pd.read_csv(filepath)
        print(f"Processing FOOD-DATA-GROUP{group_num}: {len(df)} foods (using 100 samples)")
        
        # Sample 100 diverse foods from each group
        sample_df = df.sample(n=min(100, len(df)), random_state=42)
        
        for _, row in sample_df.iterrows():
            food_name = row['food']
            
            # Build comprehensive nutrition content
            content_parts = [
                f"{food_name} is a nutritious food with the following profile:",
                f"",
                f"Macronutrients (per 100g):",
                f"- Calories: {row['Caloric Value']} kcal",
                f"- Protein: {row['Protein']:.1f}g",
                f"- Carbohydrates: {row['Carbohydrates']:.1f}g",
                f"- Fats: {row['Fat']:.1f}g (Saturated: {row['Saturated Fats']:.1f}g, Monounsaturated: {row['Monounsaturated Fats']:.1f}g, Polyunsaturated: {row['Polyunsaturated Fats']:.1f}g)",
                f"- Dietary Fiber: {row['Dietary Fiber']:.1f}g",
                f"- Sugars: {row['Sugars']:.1f}g",
                f"",
                f"Micronutrients:",
                f"- Vitamins: A ({row['Vitamin A']:.3f}mg), B1 ({row['Vitamin B1']:.3f}mg), B2 ({row['Vitamin B2']:.3f}mg), B3 ({row['Vitamin B3']:.3f}mg), B6 ({row['Vitamin B6']:.3f}mg), B12 ({row['Vitamin B12']:.3f}mg), C ({row['Vitamin C']:.3f}mg), D ({row['Vitamin D']:.3f}mg), E ({row['Vitamin E']:.3f}mg), K ({row['Vitamin K']:.3f}mg)",
                f"- Minerals: Calcium ({row['Calcium']:.1f}mg), Iron ({row['Iron']:.3f}mg), Magnesium ({row['Magnesium']:.1f}mg), Phosphorus ({row['Phosphorus']:.1f}mg), Potassium ({row['Potassium']:.1f}mg), Sodium ({row['Sodium']:.3f}mg), Zinc ({row['Zinc']:.3f}mg)",
                f"- Other: Cholesterol ({row['Cholesterol']:.1f}mg), Water ({row['Water']:.1f}g)",
                f"",
                f"Nutrition Density Score: {row['Nutrition Density']:.2f}"
            ]
            
            docs.append({
                "id": f"food_{doc_id}",
                "category": "food_nutrition_facts",
                "title": f"Nutritional profile of {food_name}",
                "content": "\n".join(content_parts),
                "source": f"FINAL FOOD DATASET Group {group_num}"
            })
            doc_id += 1
    
    return docs

def create_meal_plan_docs(base_path: str) -> List[Dict]:
    """Convert Food Nutrition Dataset (meals & macros) to knowledge base"""
    docs = []
    doc_id = 2000
    
    filepath = os.path.join(base_path, "Food Nutrition Dataset/detailed_meals_macros_CLEANED.csv")
    if not os.path.exists(filepath):
        return docs
    
    df = pd.read_csv(filepath)
    print(f"Processing detailed_meals_macros: {len(df)} meal plans (using 100 samples)")
    
    # Sample representative meal plans (avoid overwhelming the knowledge base)
    # Use 100 diverse samples
    sample_df = df.sample(n=min(100, len(df)), random_state=42)
    
    for _, row in sample_df.iterrows():
        # Build meal plan content
        content_parts = [
            f"Personalized meal plan for {row['Gender']} aged {row['Ages']}, {row['Height']}cm, {row['Weight']}kg:",
            f"",
            f"Profile:",
            f"- Activity Level: {row['Activity Level']}",
            f"- Dietary Preference: {row['Dietary Preference']}",
            f"- Health Conditions: {row['Disease']}",
            f"- Daily Calorie Target: {row['Daily Calorie Target']} kcal",
            f"",
            f"Daily Macros:",
            f"- Protein: {row['Protein']}g",
            f"- Carbohydrates: {row['Carbohydrates']}g",
            f"- Fats: {row['Fat']}g",
            f"- Fiber: {row['Fiber']}g",
            f"",
            f"Meal Suggestions:",
            f"",
            f"Breakfast: {row['Breakfast Suggestion']}",
            f"- Calories: {row['Breakfast Calories']:.0f} kcal",
            f"- Protein: {row['Breakfast Protein']:.0f}g, Carbs: {row['Breakfast Carbohydrates']:.0f}g, Fats: {row['Breakfast Fats']:.0f}g",
            f"",
            f"Lunch: {row['Lunch Suggestion']}",
            f"- Calories: {row['Lunch Calories']:.0f} kcal",
            f"- Protein: {row['Lunch Protein']:.0f}g, Carbs: {row['Lunch Carbohydrates']:.0f}g, Fats: {row['Lunch Fats']:.0f}g",
            f"",
            f"Dinner: {row['Dinner Suggestion']}",
            f"- Calories: {row['Dinner Calories']:.0f} kcal",
            f"- Protein: {row['Dinner Protein.1']:.0f}g, Carbs: {row['Dinner Carbohydrates.1']:.0f}g, Fats: {row['Dinner Fats']:.0f}g",
            f"",
            f"Snacks: {row['Snack Suggestion']}",
            f"- Calories: {row['Snacks Calories']} kcal",
            f"- Protein: {row['Snacks Protein']}g, Carbs: {row['Snacks Carbohydrates']}g, Fats: {row['Snacks Fats']}g"
        ]
        
        # Create category based on dietary preference and health condition
        category = f"{row['Dietary Preference'].lower().replace(' ', '_')}_meal_plans"
        
        docs.append({
            "id": f"meal_{doc_id}",
            "category": category,
            "title": f"Meal plan for {row['Dietary Preference']} diet with {row['Disease']}",
            "content": "\n".join(content_parts),
            "source": "Food Nutrition Dataset - Kaggle"
        })
        doc_id += 1
    
    return docs

def create_diet_recommendation_docs(base_path: str) -> List[Dict]:
    """Convert Diet Recommendation Dataset to knowledge base"""
    docs = []
    doc_id = 3000
    
    filepath = os.path.join(base_path, "Diet Recommendation Dataset/diet_recommendations_dataset.csv")
    if not os.path.exists(filepath):
        return docs
    
    df = pd.read_csv(filepath)
    print(f"Processing diet_recommendations: {len(df)} recommendations (using 100 samples)")
    
    # Sample 100 diverse recommendations
    sample_df = df.sample(n=min(100, len(df)), random_state=42)
    
    for _, row in sample_df.iterrows():
        # Handle NaN values
        disease_type = row['Disease_Type'] if pd.notna(row['Disease_Type']) else "General Health"
        severity = row['Severity'] if pd.notna(row['Severity']) else "N/A"
        allergies = row['Allergies'] if pd.notna(row['Allergies']) else "None"
        restrictions = row['Dietary_Restrictions'] if pd.notna(row['Dietary_Restrictions']) else "None"
        
        content_parts = [
            f"Diet recommendation for patient with {disease_type} ({severity} severity):",
            f"",
            f"Patient Profile:",
            f"- Age: {row['Age']}, Gender: {row['Gender']}",
            f"- Height: {row['Height_cm']}cm, Weight: {row['Weight_kg']}kg, BMI: {row['BMI']:.1f}",
            f"- Physical Activity: {row['Physical_Activity_Level']}",
            f"- Weekly Exercise: {row['Weekly_Exercise_Hours']:.1f} hours",
            f"",
            f"Health Metrics:",
            f"- Daily Caloric Intake: {row['Daily_Caloric_Intake']} kcal",
            f"- Cholesterol: {row['Cholesterol_mg/dL']:.1f} mg/dL",
            f"- Blood Pressure: {row['Blood_Pressure_mmHg']} mmHg",
            f"- Glucose: {row['Glucose_mg/dL']:.1f} mg/dL",
            f"",
            f"Dietary Considerations:",
            f"- Allergies: {allergies}",
            f"- Preferred Cuisine: {row['Preferred_Cuisine']}",
            f"- Restrictions: {restrictions}",
            f"",
            f"Recommended Diet Plan: {row['Diet_Recommendation']}",
            f"",
            f"This recommendation is based on the patient's disease type, activity level, and health metrics. The {row['Diet_Recommendation']} approach is suitable for managing {disease_type} while considering {severity} severity."
        ]
        
        # Category based on disease type
        category = f"diet_{disease_type.lower().replace(' ', '_')}"
        
        docs.append({
            "id": f"recommendation_{doc_id}",
            "category": category,
            "title": f"{row['Diet_Recommendation']} diet for {disease_type}",
            "content": "\n".join(content_parts),
            "source": "Diet Recommendation Dataset - Kaggle"
        })
        doc_id += 1
    
    return docs

def create_personalized_recommendations_docs(base_path: str) -> List[Dict]:
    """Convert Personalized Diet Recommendations to knowledge base"""
    docs = []
    doc_id = 4000
    
    filepath = os.path.join(base_path, "dataset_4/Personalized_Diet_Recommendations.csv")
    if not os.path.exists(filepath):
        return docs
    
    df = pd.read_csv(filepath)
    print(f"Processing personalized_diet_recommendations: {len(df)} recommendations (using 50 samples)")
    
    # Sample 50 diverse recommendations
    sample_df = df.sample(n=min(50, len(df)), random_state=42)
    
    for _, row in sample_df.iterrows():
        chronic = row['Chronic_Disease'] if pd.notna(row['Chronic_Disease']) else "None"
        allergies = row['Allergies'] if pd.notna(row['Allergies']) else "None"
        aversions = row['Food_Aversions'] if pd.notna(row['Food_Aversions']) else "None"
        
        content_parts = [
            f"Personalized nutrition recommendation for {row['Age']}-year-old {row['Gender']}:",
            f"",
            f"Physical Profile:",
            f"- Height: {row['Height_cm']}cm, Weight: {row['Weight_kg']}kg, BMI: {row['BMI']:.2f}",
            f"- Daily Steps: {row['Daily_Steps']}, Exercise Frequency: {row['Exercise_Frequency']} times/week",
            f"- Sleep: {row['Sleep_Hours']:.1f} hours/night",
            f"",
            f"Health Status:",
            f"- Chronic Disease: {chronic}",
            f"- Blood Pressure: {row['Blood_Pressure_Systolic']}/{row['Blood_Pressure_Diastolic']} mmHg",
            f"- Cholesterol: {row['Cholesterol_Level']} mg/dL",
            f"- Blood Sugar: {row['Blood_Sugar_Level']} mg/dL",
            f"- Genetic Risk: {row['Genetic_Risk_Factor']}",
            f"",
            f"Lifestyle Factors:",
            f"- Alcohol: {row['Alcohol_Consumption']}, Smoking: {row['Smoking_Habit']}",
            f"- Dietary Habits: {row['Dietary_Habits']}",
            f"- Preferred Cuisine: {row['Preferred_Cuisine']}",
            f"- Allergies: {allergies}",
            f"- Food Aversions: {aversions}",
            f"",
            f"Current Intake:",
            f"- Calories: {row['Caloric_Intake']} kcal",
            f"- Protein: {row['Protein_Intake']}g",
            f"- Carbohydrates: {row['Carbohydrate_Intake']}g",
            f"- Fats: {row['Fat_Intake']}g",
            f"",
            f"Recommended Daily Intake:",
            f"- Calories: {row['Recommended_Calories']} kcal",
            f"- Protein: {row['Recommended_Protein']}g",
            f"- Carbohydrates: {row['Recommended_Carbs']}g",
            f"- Fats: {row['Recommended_Fats']}g",
            f"",
            f"Recommended Meal Plan: {row['Recommended_Meal_Plan']}",
            f"",
            f"This personalized plan considers your {row['Dietary_Habits']} dietary habits, {row['Preferred_Cuisine']} cuisine preference, and is designed to support your health goals while managing {chronic}."
        ]
        
        category = f"{row['Dietary_Habits'].lower().replace(' ', '_')}_personalized"
        
        docs.append({
            "id": f"personalized_{doc_id}",
            "category": category,
            "title": f"{row['Recommended_Meal_Plan']} for {row['Dietary_Habits']} {row['Gender']}",
            "content": "\n".join(content_parts),
            "source": "Personalized Diet Recommendations - Kaggle"
        })
        doc_id += 1
    
    return docs

def main():
    """Convert all Kaggle datasets to RAG knowledge base"""
    base_path = "data/raw"
    
    print("="*80)
    print("Converting Kaggle Nutrition Datasets to RAG Knowledge Base")
    print("="*80)
    
    all_docs = []
    
    # Process all datasets
    print("\n1. Processing FINAL FOOD DATASET (5 groups)...")
    all_docs.extend(create_food_nutrition_docs(base_path))
    
    print("\n2. Processing Food Nutrition Dataset (meals & macros)...")
    all_docs.extend(create_meal_plan_docs(base_path))
    
    print("\n3. Processing Diet Recommendation Dataset...")
    all_docs.extend(create_diet_recommendation_docs(base_path))
    
    print("\n4. Processing Personalized Diet Recommendations...")
    all_docs.extend(create_personalized_recommendations_docs(base_path))
    
    # Save to JSON
    output_path = "knowledge_base/kaggle_nutrition.json"
    os.makedirs("knowledge_base", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print(f"✅ Knowledge base created successfully!")
    print(f"Total documents: {len(all_docs)}")
    print(f"Output file: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print("="*80)
    
    # Print category breakdown
    print("\nDocument categories:")
    categories = {}
    for doc in all_docs:
        cat = doc['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count} documents")

if __name__ == "__main__":
    main()
