import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please run model_training.py first.")
        return None, None

@st.cache_data
def load_results():
    """Load model comparison results"""
    try:
        return pd.read_csv('model_results.csv', index_col=0)
    except FileNotFoundError:
        return None

def predict_batch_fraud(model, scaler, df):
    """Make batch fraud predictions"""
    # Ensure all required columns are present
    required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        return None
    
    # Select and reorder columns to match training order
    df_features = df[required_cols].copy()
    
    # Convert to numpy array to avoid feature name issues
    features_array = df_features.values  # Shape: (n_samples, 30)
    
    # Scale only Time (index 0) and Amount (index 1) columns
    # Scaler expects [Amount, Time] order
    time_amount_data = features_array[:, [1, 0]]  # Extract [Amount, Time]
    time_amount_scaled = scaler.transform(time_amount_data)
    
    # Replace the scaled values back
    features_scaled = features_array.copy()
    features_scaled[:, 0] = time_amount_scaled[:, 1]  # Time (scaled)
    features_scaled[:, 1] = time_amount_scaled[:, 0]  # Amount (scaled)
    
    # Make predictions using numpy array
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[:, 1]
    
    # Create results dataframe with lower threshold for fraud detection
    results_df = df_features.copy()
    fraud_threshold = 0.1  # Lower threshold for fraud detection
    results_df['Prediction'] = ['Fraud' if prob > fraud_threshold else 'Legitimate' for prob in probabilities]
    results_df['Fraud_Probability'] = probabilities
    results_df['Risk_Level'] = ['High' if p > 0.3 else 'Medium' if p > 0.1 else 'Low' for p in probabilities]
    
    return results_df

def create_sample_csv():
    """Create a sample CSV for download"""
    sample_data = {
        'Time': [0, 1, 2],
        'Amount': [149.62, 2.69, 378.66]
    }
    
    # Add V1-V28 with sample values
    for i in range(1, 29):
        sample_data[f'V{i}'] = [0.0, 0.1, -0.1]
    
    return pd.DataFrame(sample_data)

def predict_single_fraud(model, scaler, time, amount, v_features):
    """Make single fraud prediction"""
    # Create feature array [Time, Amount, V1, V2, ..., V28]
    features = [time, amount] + [v_features[f'V{i}'] for i in range(1, 29)]
    features_array = np.array(features).reshape(1, -1)
    
    # Scale Time and Amount - scaler expects [Amount, Time] order
    time_amount_scaled = scaler.transform([[amount, time]])[0]
    
    # Replace with scaled values
    features_scaled = features_array.copy()
    features_scaled[0, 0] = time_amount_scaled[1]  # Time (scaled)
    features_scaled[0, 1] = time_amount_scaled[0]  # Amount (scaled)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return prediction, probability

def create_model_comparison_chart(results_df):
    """Create model comparison chart"""
    if results_df is not None:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for metric, (row, col) in zip(metrics, positions):
            fig.add_trace(
                go.Bar(x=results_df.index, y=results_df[metric], name=metric.title()),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=False, title_text="Model Performance Comparison")
        return fig
    return None

def main():
    """Main Streamlit app"""
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("---")
    
    # Load model and scaler
    model, scaler = load_model()
    
    if model is None or scaler is None:
        st.stop()
    
    # Sidebar with model info
    st.sidebar.header("📊 Model Information")
    st.sidebar.info(f"**Model Type:** {type(model).__name__}")
    st.sidebar.info("**Features:** 30 (Time, Amount, V1-V28)")
    
    # Load and display model results
    results_df = load_results()
    if results_df is not None:
        st.sidebar.subheader("Model Performance")
        best_model = results_df.loc[results_df['f1_score'].idxmax()]
        st.sidebar.metric("Best F1-Score", f"{best_model['f1_score']:.4f}")
        st.sidebar.metric("ROC-AUC", f"{best_model['roc_auc']:.4f}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📁 Batch Upload", "✏️ Manual Input", "📈 Model Comparison"])
    
    with tab1:
        st.subheader("Upload CSV File for Batch Fraud Detection")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.info("**Required CSV Format:**\n- Time, Amount, V1, V2, ..., V28 (30 columns total)\n- No 'Class' column needed")
            
            # Sample CSV download
            sample_df = create_sample_csv()
            csv_sample = sample_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv_sample,
                file_name="sample_transactions.csv",
                mime="text/csv"
            )
        
        with col2:
            # File upload
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type="csv",
                help="Upload a CSV file with transaction data"
            )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! {len(df)} transactions loaded.")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Predict button
                if st.button("🔍 Analyze All Transactions", type="primary"):
                    with st.spinner("Analyzing transactions..."):
                        results = predict_batch_fraud(model, scaler, df)
                    
                    if results is not None:
                        st.subheader("Fraud Detection Results")
                        
                        # Summary metrics
                        fraud_count = len(results[results['Prediction'] == 'Fraud'])
                        total_count = len(results)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Transactions", total_count)
                        with col2:
                            st.metric("Fraudulent", fraud_count)
                        with col3:
                            st.metric("Legitimate", total_count - fraud_count)
                        with col4:
                            st.metric("Fraud Rate", f"{fraud_count/total_count*100:.1f}%")
                        
                        # Results table
                        st.dataframe(results, use_container_width=True)
                        
                        # Download results
                        csv_results = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv_results,
                            file_name="fraud_detection_results.csv",
                            mime="text/csv"
                        )
                        
                        # Visualization
                        fig_dist = px.histogram(
                            results, x='Fraud_Probability', color='Prediction',
                            title="Fraud Probability Distribution",
                            nbins=20
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        st.subheader("Manual Transaction Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            time = st.number_input("Time", value=0.0, help="Time elapsed since first transaction")
            amount = st.number_input("Amount", value=100.0, min_value=0.0, help="Transaction amount")
        
        with col2:
            st.write("**PCA Features (V1-V28)**")
        
        # Create input fields for V1-V28 features
        v_features = {}
        cols = st.columns(4)
        for i in range(1, 29):
            col_idx = (i-1) % 4
            with cols[col_idx]:
                v_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.6f")
        
        # Predict button
        if st.button("🔍 Predict Single Transaction", type="primary"):
            prediction, probability = predict_single_fraud(model, scaler, time, amount, v_features)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Prediction Result")
                
                # Use lower threshold for fraud detection
                fraud_threshold = 0.1
                is_fraud = probability[1] > fraud_threshold
                
                if is_fraud:
                    st.error("🚨 **FRAUDULENT TRANSACTION**")
                    st.metric("Fraud Probability", f"{probability[1]:.2%}")
                else:
                    st.success("✅ **LEGITIMATE TRANSACTION**")
                    st.metric("Fraud Probability", f"{probability[1]:.2%}")
            
            with col2:
                # Probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability[1] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fraud Risk %"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "red" if is_fraud else "green"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
    
    with tab3:
        st.subheader("Model Performance Comparison")
        
        if results_df is not None:
            # Display results table
            st.dataframe(results_df.round(4), use_container_width=True)
            
            # Display comparison chart
            fig_comparison = create_model_comparison_chart(results_df)
            if fig_comparison:
                st.plotly_chart(fig_comparison, use_container_width=True)
        else:
            st.warning("Model comparison results not available. Please run model_training.py first.")

if __name__ == "__main__":
    main()