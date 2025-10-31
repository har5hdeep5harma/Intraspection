import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the anxiety dataset"""
    print("Loading dataset...")
    df = pd.read_csv('data/anxiety_dataset.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Handle categorical variables
    print("\nEncoding categorical variables...")
    
    # Gender encoding
    gender_map = {'Male': 1, 'Female': 2, 'Other': 0}
    df['Gender'] = df['Gender'].map(gender_map)
    
    # Occupation encoding
    occupation_encoder = LabelEncoder()
    df['Occupation'] = occupation_encoder.fit_transform(df['Occupation'])
    
    # Yes/No encoding
    yes_no_cols = ['Smoking', 'Family History of Anxiety', 'Dizziness', 'Medication', 'Recent Major Life Event']
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Features and target
    X = df.drop('Anxiety Level (1-10)', axis=1)
    y = df['Anxiety Level (1-10)']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature names: {X.columns.tolist()}")
    
    return X, y, occupation_encoder

def train_and_evaluate_models(X, y):
    """Train multiple models and compare performance"""
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define models
    models = {
        'random_forest': {
            'model': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
            'name': 'Random Forest',
            'description': 'Ensemble of decision trees with high accuracy and interpretability'
        },
        'xgboost': {
            'model': xgb.XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.1, random_state=42),
            'name': 'XGBoost',
            'description': 'Gradient boosting with excellent performance on structured data'
        },
        'gradient_boosting': {
            'model': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
            'name': 'Gradient Boosting',
            'description': 'Sequential ensemble method with strong predictive power'
        },
        'svr': {
            'model': SVR(kernel='rbf', C=10, gamma='scale'),
            'name': 'Support Vector Regression',
            'description': 'Kernel-based method effective for complex non-linear relationships'
        },
        'neural_network': {
            'model': MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=500, random_state=42),
            'name': 'Neural Network',
            'description': 'Deep learning approach with multiple hidden layers'
        }
    }
    
    results = {}
    trained_models = {}
    
    for key, model_config in models.items():
        print(f"\nTraining {model_config['name']}...")
        model = model_config['model']
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[key] = {
            'name': model_config['name'],
            'description': model_config['description'],
            'train_r2': round(train_r2, 4),
            'test_r2': round(test_r2, 4),
            'mae': round(test_mae, 4),
            'rmse': round(test_rmse, 4),
            'cv_score': round(cv_mean, 4),
            'cv_std': round(cv_std, 4)
        }
        
        trained_models[key] = model
        
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    return trained_models, results

def save_models(models, model_info):
    """Save trained models and their metadata"""
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    os.makedirs('models', exist_ok=True)
    
    for key, model in models.items():
        model_path = os.path.join('models', f'{key}.pkl')
        joblib.dump(model, model_path)
        print(f"Saved {key} to {model_path}")
    
    # Save model info
    info_path = os.path.join('models', 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"\nSaved model information to {info_path}")

def print_summary(results):
    """Print a summary of all models"""
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    # Sort by test R² score
    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Model':<30} {'Test R²':<10} {'MAE':<10} {'CV Score':<10}")
    print("-" * 65)
    
    for rank, (key, info) in enumerate(sorted_models, 1):
        print(f"{rank:<5} {info['name']:<30} {info['test_r2']:<10} {info['mae']:<10} {info['cv_score']:<10}")
    
    best_model = sorted_models[0]
    print(f"\n🏆 Best Model: {best_model[1]['name']}")
    print(f"   Test R² Score: {best_model[1]['test_r2']}")
    print(f"   Mean Absolute Error: {best_model[1]['mae']}")

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("SOCIAL ANXIETY PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Load and preprocess data
    X, y, occupation_encoder = load_and_preprocess_data()
    
    # Train models
    trained_models, results = train_and_evaluate_models(X, y)
    
    # Save models
    save_models(trained_models, results)
    
    # Print summary
    print_summary(results)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nYou can now run the Flask app with: python app.py")

if __name__ == '__main__':
    main()
