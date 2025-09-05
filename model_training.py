import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import get_processed_data

def train_models(X_train, y_train):
    """Train multiple classification models"""
    import time
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
    }
    
    trained_models = {}
    for name, model in models.items():
        start_time = time.time()
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        end_time = time.time()
        print(f"  ✓ {name} completed in {end_time - start_time:.1f} seconds")
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate all models and return performance metrics"""
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
    
    return results

def select_best_model(results):
    """Select the best model based on F1-score"""
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    return best_model_name

def save_model(model, filename='best_model.pkl'):
    """Save the best model"""
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def print_results(results):
    """Print model evaluation results"""
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    
    df_results = pd.DataFrame(results).T
    df_results = df_results.round(4)
    print(df_results)
    
    return df_results

def main():
    """Main training pipeline"""
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = get_processed_data()
    
    print("\nTraining models...")
    models = train_models(X_train, y_train)
    
    print("\nEvaluating models...")
    results = evaluate_models(models, X_test, y_test)
    
    # Print results
    df_results = print_results(results)
    
    # Select and save best model
    best_model_name = select_best_model(results)
    best_model = models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best F1-score: {results[best_model_name]['f1_score']:.4f}")
    
    # Save the best model
    save_model(best_model)
    
    # Save results for Streamlit app
    df_results.to_csv('model_results.csv')
    
    return best_model, results

if __name__ == "__main__":
    best_model, results = main()