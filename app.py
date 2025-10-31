from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json

app = Flask(__name__)

# Load trained models
MODELS_DIR = 'models'
models = {}
model_info = {}

def load_models():
    """Load all trained models and their metadata"""
    global models, model_info
    if os.path.exists(MODELS_DIR):
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '')
            models[model_name] = joblib.load(os.path.join(MODELS_DIR, model_file))
        
        # Load model info if exists
        info_path = os.path.join(MODELS_DIR, 'model_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = json.load(f)

# Load models on startup
try:
    load_models()
    if not models:
        print("   WARNING: No models found in models/ directory")
        print("   Please run: python train_models.py")
        print("   Or ensure trained .pkl files are deployed with your app")
except Exception as e:
    print(f"   ERROR loading models: {e}")
    print("   Please run: python train_models.py")
    print("   Or ensure trained .pkl files are deployed with your app")

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    """Anxiety Prediction page"""
    return render_template('predict.html', models=model_info)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for anxiety level prediction"""
    try:
        # Check if models are loaded
        if not models:
            return jsonify({
                'error': 'Models not loaded. Please ensure trained models are available.',
                'details': 'Contact administrator or run train_models.py'
            }), 503
        
        data = request.json
        model_name = data.get('model', 'random_forest')
        
        if model_name not in models:
            return jsonify({'error': 'Invalid model selected'}), 400
        
        # Extract features in the correct order
        features = [
            float(data['age']),
            1 if data['gender'] == 'Male' else (2 if data['gender'] == 'Female' else 0),
            float(data['occupation_encoded']),
            float(data['sleep_hours']),
            float(data['physical_activity']),
            float(data['caffeine_intake']),
            float(data['alcohol_consumption']),
            1 if data['smoking'] == 'Yes' else 0,
            1 if data['family_history'] == 'Yes' else 0,
            float(data['stress_level']),
            float(data['heart_rate']),
            float(data['breathing_rate']),
            float(data['sweating_level']),
            1 if data['dizziness'] == 'Yes' else 0,
            1 if data['medication'] == 'Yes' else 0,
            float(data['therapy_sessions']),
            1 if data['recent_life_event'] == 'Yes' else 0,
            float(data['diet_quality'])
        ]
        
        # Make prediction
        model = models[model_name]
        prediction = model.predict([features])[0]
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([features])[0]
            confidence = float(max(proba)) * 100
        else:
            confidence = None
        
        # Determine anxiety level category
        if prediction <= 3:
            category = "Low"
            color = "#10b981"
        elif prediction <= 6:
            category = "Moderate"
            color = "#f59e0b"
        else:
            category = "High"
            color = "#ef4444"
        
        return jsonify({
            'prediction': round(float(prediction), 2),
            'category': category,
            'color': color,
            'confidence': round(confidence, 2) if confidence else None,
            'model_used': model_name
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/disparity')
def disparity_page():
    """Disparity Level Calculator page"""
    return render_template('disparity.html')

@app.route('/api/disparity', methods=['POST'])
def calculate_disparity():
    """API endpoint for disparity coefficient calculation"""
    try:
        data = request.json
        
        # Extract current state scores
        current_scores = {
            'intelligence': int(data.get('current_intelligence', 0)),
            'knowledge': int(data.get('current_knowledge', 0)),
            'wealth': int(data.get('current_wealth', 0)),
            'sex': int(data.get('current_sex', 0)),
            'health': int(data.get('current_health', 0)),
            'motivation': int(data.get('current_motivation', 0)),
            'education': int(data.get('current_education', 0)),
            'special_skills': int(data.get('current_special_skills', 0)),
            'attractiveness': int(data.get('current_attractiveness', 0)),
            'social_skills': int(data.get('current_social_skills', 0)),
            'anxiety_resistance': int(data.get('current_anxiety_resistance', 0)),
            'depression_resistance': int(data.get('current_depression_resistance', 0)),
            'power': int(data.get('current_power', 0)),
            'enlightened': int(data.get('current_enlightened', 0))
        }
        
        # Extract desired state scores
        desired_scores = {
            'intelligence': int(data.get('desired_intelligence', 0)),
            'knowledge': int(data.get('desired_knowledge', 0)),
            'wealth': int(data.get('desired_wealth', 0)),
            'sex': int(data.get('desired_sex', 0)),
            'health': int(data.get('desired_health', 0)),
            'motivation': int(data.get('desired_motivation', 0)),
            'education': int(data.get('desired_education', 0)),
            'special_skills': int(data.get('desired_special_skills', 0)),
            'attractiveness': int(data.get('desired_attractiveness', 0)),
            'social_skills': int(data.get('desired_social_skills', 0)),
            'anxiety_resistance': int(data.get('desired_anxiety_resistance', 0)),
            'depression_resistance': int(data.get('desired_depression_resistance', 0)),
            'power': int(data.get('desired_power', 0)),
            'enlightened': int(data.get('desired_enlightened', 0))
        }
        
        # Calculate averages
        num_questions = 14
        current_total = sum(current_scores.values())
        desired_total = sum(desired_scores.values())
        current_average = current_total / num_questions
        desired_average = desired_total / num_questions
        
        # Calculate Disparity Coefficient (desired average - current average)
        disparity_coefficient = desired_average - current_average
        
        # Calculate individual disparities for detailed breakdown
        individual_disparities = {}
        for key in current_scores.keys():
            individual_disparities[key] = desired_scores[key] - current_scores[key]
        
        # Determine interpretation based on coefficient value
        if disparity_coefficient <= 0.5:
            interpretation = "Very Low Disparity"
            description = "Your current state aligns well with your desires. You have a strong sense of contentment and self-acceptance."
            color = "#10b981"
            advice = "Continue maintaining this balance. Focus on sustaining your current positive state."
        elif disparity_coefficient <= 1.5:
            interpretation = "Low Disparity"
            description = "Minor gaps exist between your current state and desires. You're generally satisfied but see room for growth."
            color = "#3b82f6"
            advice = "Small, consistent improvements can help close these gaps. Set achievable short-term goals."
        elif disparity_coefficient <= 2.5:
            interpretation = "Moderate Disparity"
            description = "Noticeable differences between current and desired states. This may contribute to motivation for change or mild stress."
            color = "#f59e0b"
            advice = "Break down your goals into manageable steps. Consider seeking support or resources to bridge the gaps."
        elif disparity_coefficient <= 3.5:
            interpretation = "High Disparity"
            description = "Significant gaps between where you are and where you want to be. This disparity may be causing considerable stress or despair."
            color = "#ef4444"
            advice = "Professional guidance may help. Focus on the most important areas first and set realistic expectations."
        else:
            interpretation = "Very High Disparity"
            description = "Substantial disconnect between current reality and desires. This level of disparity often correlates with feelings of despair and requires attention."
            color = "#dc2626"
            advice = "Strongly consider professional support. Re-evaluate whether your desired state is realistic and adjust if needed."
        
        # Find top 3 areas with highest positive disparity (where you want to improve most)
        sorted_disparities = sorted(individual_disparities.items(), key=lambda x: x[1], reverse=True)
        top_areas = []
        for area, gap in sorted_disparities[:3]:
            if gap > 0:
                top_areas.append({
                    'name': area.replace('_', ' ').title(),
                    'current': current_scores[area],
                    'desired': desired_scores[area],
                    'gap': gap
                })
        
        # Find areas where you're already at or above desired level
        strong_areas = []
        for area, gap in sorted_disparities:
            if gap <= 0:
                strong_areas.append({
                    'name': area.replace('_', ' ').title(),
                    'current': current_scores[area],
                    'desired': desired_scores[area]
                })
        
        return jsonify({
            'success': True,
            'current_total': current_total,
            'desired_total': desired_total,
            'current_average': round(current_average, 2),
            'desired_average': round(desired_average, 2),
            'disparity_coefficient': round(disparity_coefficient, 2),
            'interpretation': interpretation,
            'description': description,
            'advice': advice,
            'color': color,
            'top_areas': top_areas,
            'strong_areas': strong_areas[:3] if strong_areas else [],
            'individual_disparities': {k.replace('_', ' ').title(): v for k, v in individual_disparities.items()}
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/profiling')
def profiling_page():
    """Psychological Profiling page"""
    return render_template('profiling.html')

@app.route('/api/profiling', methods=['POST'])
def calculate_profile():
    """API endpoint for psychological profiling based on W.H. Sheldon's somatotype theory"""
    try:
        data = request.json
        
        # Initialize scores for each temperament type
        scores = {
            'viscerotonic': 0,  # A scores (Endomorphy)
            'somatotonic': 0,   # B scores (Mesomorphy)
            'cerebrotonic': 0   # C scores (Ectomorphy)
        }
        
        # Collect all scores from 20 sets
        for i in range(1, 21):
            set_scores = {
                'a': int(data.get(f'set{i}_a', 0)),
                'b': int(data.get(f'set{i}_b', 0)),
                'c': int(data.get(f'set{i}_c', 0))
            }
            
            scores['viscerotonic'] += set_scores['a']
            scores['somatotonic'] += set_scores['b']
            scores['cerebrotonic'] += set_scores['c']
        
        # Calculate percentages
        total_score = sum(scores.values())
        if total_score == 0:
            return jsonify({'success': False, 'error': 'Please complete all assessments'}), 400
        
        percentages = {
            'viscerotonic': round((scores['viscerotonic'] / total_score) * 100, 1),
            'somatotonic': round((scores['somatotonic'] / total_score) * 100, 1),
            'cerebrotonic': round((scores['cerebrotonic'] / total_score) * 100, 1)
        }
        
        # Determine dominant, secondary, and weakest traits
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        dominant = sorted_scores[0]
        secondary = sorted_scores[1]
        weakest = sorted_scores[2]
        
        # Profile descriptions
        profiles = {
            'viscerotonic': {
                'name': 'Viscerotonic (Comfort-Oriented)',
                'color': '#10b981',
                'icon': '🤗',
                'description': 'You have a warm, sociable, and comfort-seeking temperament. You value relationships, enjoy social gatherings, and find pleasure in physical comfort and relaxation.',
                'strengths': [
                    'Warm and friendly personality',
                    'Excellent at building relationships',
                    'Creates comfortable social environments',
                    'Good at reading emotional atmospheres',
                    'Natural host and entertainer'
                ],
                'weaknesses': [
                    'May avoid confrontation excessively',
                    'Can struggle with high-stress situations',
                    'May overindulge in comfort activities',
                    'Difficulty saying no to others',
                    'May need external validation'
                ],
                'persuasion': 'Responds best to warmth, connection, and appeals to comfort and harmony. Use relationship-building and emotional connection in communication.'
            },
            'somatotonic': {
                'name': 'Somatotonic (Action-Oriented)',
                'color': '#f59e0b',
                'icon': '⚡',
                'description': 'You have an energetic, assertive, and competitive temperament. You thrive on physical activity, challenges, and achieving goals. You are action-oriented and direct.',
                'strengths': [
                    'High energy and motivation',
                    'Natural leadership abilities',
                    'Courage and boldness',
                    'Goal-oriented and competitive',
                    'Handles physical challenges well'
                ],
                'weaknesses': [
                    'May be too aggressive or domineering',
                    'Can lack sensitivity to others\' feelings',
                    'May take unnecessary risks',
                    'Difficulty with patience and stillness',
                    'Can be insensitive to pain (own and others)'
                ],
                'persuasion': 'Responds best to challenges, achievements, and urgency. Use competition, goals, and action-oriented language to motivate.'
            },
            'cerebrotonic': {
                'name': 'Cerebrotonic (Intellect-Oriented)',
                'color': '#6366f1',
                'icon': '🧠',
                'description': 'You have a thoughtful, introspective, and analytical temperament. You value privacy, intellectual pursuits, and careful consideration. You are sensitive and prefer depth over breadth.',
                'strengths': [
                    'Deep analytical thinking',
                    'Highly observant and perceptive',
                    'Values knowledge and understanding',
                    'Self-reflective and introspective',
                    'Good at planning and strategizing'
                ],
                'weaknesses': [
                    'May overthink situations',
                    'Can be socially inhibited',
                    'Hypersensitive to stimulation',
                    'May struggle with spontaneity',
                    'Tendency toward anxiety and worry'
                ],
                'persuasion': 'Responds best to logic, subtlety, and intellectual appeal. Provide alone time, minimal sensory stimulation, and well-reasoned arguments.'
            }
        }
        
        # Generate recommendations based on profile
        recommendations = []
        
        # Dominant trait recommendations
        if dominant[0] == 'viscerotonic':
            recommendations.extend([
                'Nurture your social connections - they are your strength',
                'Create comfortable environments for productivity',
                'Practice setting boundaries while maintaining relationships',
                'Use your warmth to build strong professional networks'
            ])
        elif dominant[0] == 'somatotonic':
            recommendations.extend([
                'Channel your energy into structured physical activities',
                'Set clear, challenging goals to maintain motivation',
                'Practice empathy and active listening',
                'Balance action with reflection and planning'
            ])
        else:  # cerebrotonic
            recommendations.extend([
                'Honor your need for solitude and quiet time',
                'Use your analytical skills in problem-solving',
                'Practice gradual exposure to social situations',
                'Manage sensory input to prevent overstimulation'
            ])
        
        # Add balance recommendations
        if dominant[1] - weakest[1] > 40:
            recommendations.append(f'Work on developing your {weakest[0].replace("otonic", "")} traits for better balance')
        
        # Stress management based on dominant type
        stress_management = {
            'viscerotonic': 'Seek support from friends and family when stressed. Social connection helps you recharge.',
            'somatotonic': 'Engage in physical activity when stressed. Movement and action help you process emotions.',
            'cerebrotonic': 'Take time alone to think and process when stressed. Solitude helps you regain clarity.'
        }
        
        return jsonify({
            'success': True,
            'scores': scores,
            'percentages': percentages,
            'dominant': {
                'type': dominant[0],
                'score': dominant[1],
                'percentage': percentages[dominant[0]],
                'profile': profiles[dominant[0]]
            },
            'secondary': {
                'type': secondary[0],
                'score': secondary[1],
                'percentage': percentages[secondary[0]],
                'profile': profiles[secondary[0]]
            },
            'weakest': {
                'type': weakest[0],
                'score': weakest[1],
                'percentage': percentages[weakest[0]],
                'profile': profiles[weakest[0]]
            },
            'recommendations': recommendations,
            'stress_management': stress_management[dominant[0]],
            'balance_score': round(100 - (abs(dominant[1] - weakest[1]) / total_score * 100), 1)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
