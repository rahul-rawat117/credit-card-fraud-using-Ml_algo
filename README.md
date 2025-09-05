# Credit Card Fraud Detection System

🚀 [Live Demo]((https://creditcardfraudrahulapex.streamlit.app/))| 📊 [GitHub Repository](https://github.com/rahul-rawat117/credit-card-fraud-using-Ml_algo))

A complete machine learning system for detecting credit card fraud with multiple models and a Streamlit web interface.

## Features

- Multiple ML Models: Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting
- Class Imbalance Handling: SMOTE oversampling technique
- Interactive Web UI: Streamlit-based interface for real-time predictions
- Model Comparison: Visual comparison of model performances
- Modular Design: Separate modules for preprocessing, training, and deployment

## Project Structure

```
├── creditcard.csv          # Dataset (Kaggle credit card fraud dataset)
├── data_preprocessing.py   # Data loading and preprocessing functions
├── model_training.py       # Model training and evaluation
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── best_model.pkl         # Trained best model (generated after training)
├── scaler.pkl            # Feature scaler (generated after training)
└── model_results.csv     # Model comparison results (generated after training)
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Models
Run the model training script to train all models and save the best one:
```bash
python model_training.py
```

This will:
- Load and preprocess the dataset
- Handle class imbalance using SMOTE
- Train 5 different ML models
- Evaluate and compare models
- Save the best performing model as `best_model.pkl`

### 2. Run the Web App
Launch the Streamlit application:
```bash
streamlit run app.py
```

The web app provides:
- Fraud Detection Tab: Input transaction features and get real-time predictions
- Model Comparison Tab: View performance metrics of all trained models
- Interactive Gauges: Visual representation of fraud probability

## Model Performance

The system trains and compares:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Dataset

The system expects a credit card fraud dataset with:
- Time: Time elapsed since first transaction
- Amount: Transaction amount
- V1-V28: PCA-transformed features
- Class: Target variable (0=Legitimate, 1=Fraud)

## Key Features

- Data Preprocessing: Automatic scaling of Amount and Time features
- Class Imbalance: SMOTE oversampling for balanced training
- Model Selection: Automatic selection of best model based on F1-score
- Real-time Prediction: Interactive web interface for fraud detection
- Visualization: Performance comparison charts and fraud probability gauges
