import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def load_data(file_path='creditcard.csv'):
    """Load the credit card dataset"""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocess the data: handle missing values, scale features"""
    # Check for missing values
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale the Amount and Time features
    scaler = StandardScaler()
    X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])
    
    # Save the scaler for later use
    joblib.dump(scaler, 'scaler.pkl')
    
    return X, y

def handle_imbalance(X, y, method='smote'):
    """Handle class imbalance using SMOTE"""
    if method == 'smote':
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def get_processed_data():
    """Main function to get processed data"""
    # Load data
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Split data first
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Handle imbalance only on training data
    X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train)
    
    print(f"Training set after SMOTE: {X_train_balanced.shape}")
    print(f"Fraud cases in balanced training set: {y_train_balanced.sum()} ({y_train_balanced.mean()*100:.2f}%)")
    
    return X_train_balanced, X_test, y_train_balanced, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_processed_data()