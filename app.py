import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import sys
import os

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import DataPreprocessor
from model_training import DiabetesPredictor

class DiabetesPredictionApp:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.predictor = DiabetesPredictor()
        self.set_page_config()
        
    def set_page_config(self):
        """Configure the Streamlit page"""
        st.set_page_config(
            page_title="Diabetes Prediction System",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def load_css(self):
        """Load custom CSS for styling"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #2E86AB;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #A23B72;
            margin-bottom: 1rem;
        }
        .prediction-box {
            padding: 2rem;
            border-radius: 10px;
            margin: 1rem 0;
            text-align: center;
        }
        .high-risk {
            background-color: #ffcccc;
            border: 2px solid #ff0000;
        }
        .low-risk {
            background-color: #ccffcc;
            border: 2px solid #00ff00;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def sidebar_inputs(self):
        """Create sidebar for user inputs"""
        st.sidebar.title("🔍 Patient Information")
        
        with st.sidebar.form("patient_data"):
            st.subheader("Enter Patient Details")
            
            pregnancies = st.slider("Pregnancies", 0, 15, 1)
            glucose = st.slider("Glucose Level (mg/dL)", 50, 200, 120)
            blood_pressure = st.slider("Blood Pressure (mm Hg)", 40, 120, 70)
            skin_thickness = st.slider("Skin Thickness (mm)", 10, 60, 20)
            insulin = st.slider("Insulin Level (mu U/ml)", 0, 300, 80)
            bmi = st.slider("BMI", 15.0, 50.0, 25.0)
            diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.08, 2.5, 0.5)
            age = st.slider("Age", 21, 80, 30)
            
            submitted = st.form_submit_button("Predict Diabetes Risk")
        
        patient_data = {
            'pregnancies': pregnancies,
            'glucose': glucose,
            'blood_pressure': blood_pressure,
            'skin_thickness': skin_thickness,
            'insulin': insulin,
            'bmi': bmi,
            'diabetes_pedigree': diabetes_pedigree,
            'age': age
        }
        
        return submitted, patient_data
    
    def display_prediction_result(self, prediction, probability):
        """Display prediction result in an attractive way"""
        risk_percentage = probability[1] * 100
        
        if prediction[0] == 1:
            st.markdown(f"""
            <div class="prediction-box high-risk">
                <h2>🚨 High Diabetes Risk</h2>
                <h3>Risk Probability: {risk_percentage:.1f}%</h3>
                <p>Please consult with a healthcare professional for further evaluation.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box low-risk">
                <h2>✅ Low Diabetes Risk</h2>
                <h3>Risk Probability: {risk_percentage:.1f}%</h3>
                <p>Maintain healthy lifestyle habits to prevent future risks.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress bar for risk visualization
        st.subheader("Risk Level Visualization")
        st.progress(probability[1])
        st.write(f"Diabetes Risk Score: {probability[1]:.3f}")
    
    def display_feature_importance(self, feature_importance):
        """Display feature importance chart"""
        if feature_importance is not None:
            st.subheader("📊 Feature Importance")
            
            fig = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title='Factors Influencing Diabetes Prediction',
                color='importance',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def display_risk_factors(self, patient_data):
        """Display risk factor analysis"""
        st.subheader("🔍 Risk Factor Analysis")
        
        risk_factors = []
        
        # Analyze each risk factor
        if patient_data['glucose'] > 140:
            risk_factors.append(f"High glucose level ({patient_data['glucose']} mg/dL)")
        if patient_data['bmi'] > 30:
            risk_factors.append(f"High BMI ({patient_data['bmi']})")
        if patient_data['age'] > 45:
            risk_factors.append(f"Age above 45 ({patient_data['age']} years)")
        if patient_data['diabetes_pedigree'] > 0.8:
            risk_factors.append("Strong family history of diabetes")
        
        if risk_factors:
            st.warning("⚠️ The following risk factors were identified:")
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.success("✅ No major risk factors identified based on input data.")
    
    def train_model_section(self):
        """Section for training the model"""
        st.sidebar.title("🤖 Model Training")
        
        if st.sidebar.button("Train New Model"):
            with st.spinner("Training high-accuracy models..."):
                # Load and preprocess data
                df = self.preprocessor.load_and_preprocess_data()
                X, y = self.preprocessor.preprocess_features(df)
                X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
                
                # Preprocess features
                self.preprocessor.fit_scaler(X_train)
                X_train_scaled = self.preprocessor.transform_features(X_train)
                X_test_scaled = self.preprocessor.transform_features(X_test)
                
                # Train models
                self.predictor.initialize_models()
                results = self.predictor.train_models(X_train_scaled, y_train, X_test_scaled, y_test)
                
                # Select best model
                best_model_name, best_results = self.predictor.select_best_model(results)
                
                # Save model and preprocessor
                self.predictor.save_model('best_diabetes_model.pkl')
                self.preprocessor.save_preprocessor('preprocessor.pkl')
                
                # Display results
                st.success(f"Model training completed! Best model: {best_model_name}")
                
                # Show model comparison
                self.display_model_comparison(results)
    
    def display_model_comparison(self, results):
        """Display model performance comparison"""
        st.subheader("📈 Model Performance Comparison")
        
        metrics_df = pd.DataFrame.from_dict(results, orient='index')
        metrics_df = metrics_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
        
        # Create comparison chart
        fig = go.Figure()
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=metrics_df.index,
                y=metrics_df[metric],
                text=metrics_df[metric].round(3)
            ))
        
        fig.update_layout(
            barmode='group',
            title='Model Performance Metrics Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Run the main application"""
        self.load_css()
        
        # Header
        st.markdown('<h1 class="main-header">🏥 Diabetes Prediction System</h1>', unsafe_allow_html=True)
        st.markdown("""
        This system uses advanced machine learning algorithms to predict the likelihood of diabetes 
        based on patient health metrics. Enter the patient information in the sidebar to get started.
        """)
        
        # Train model section in sidebar
        self.train_model_section()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Quick Prediction")
            
            # Load model and preprocessor if available
            try:
                self.predictor.load_model('best_diabetes_model.pkl')
                self.preprocessor.load_preprocessor('preprocessor.pkl')
                model_loaded = True
            except:
                model_loaded = False
                st.warning("Please train a model first using the sidebar option.")
            
            # Get user inputs
            submitted, patient_data = self.sidebar_inputs()
            
            if submitted and model_loaded:
                # Convert to DataFrame
                patient_df = pd.DataFrame([patient_data])
                
                # Preprocess and predict
                patient_scaled = self.preprocessor.transform_features(patient_df)
                prediction, probability = self.predictor.predict(patient_scaled)
                
                # Display results
                self.display_prediction_result(prediction, probability[0])
                
                # Display risk factors
                self.display_risk_factors(patient_data)
                
                # Display feature importance
                feature_names = patient_df.columns.tolist()
                feature_importance = self.predictor.get_feature_importance(feature_names)
                self.display_feature_importance(feature_importance)
        
        with col2:
            st.subheader("📋 Health Guidelines")
            st.markdown("""
            <div class="metric-card">
            <h4>Normal Ranges</h4>
            • Glucose: 70-140 mg/dL<br>
            • BMI: 18.5-24.9<br>
            • Blood Pressure: <120/80 mmHg<br>
            • Skin Thickness: 10-40 mm
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
            <h4>Prevention Tips</h4>
            • Maintain healthy weight<br>
            • Regular exercise<br>
            • Balanced diet<br>
            • Regular check-ups<br>
            • Limit sugar intake
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
            <h4>Risk Factors</h4>
            • Family history<br>
            • Overweight<br>
            • Age > 45<br>
            • High blood pressure<br>
            • Physical inactivity
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    app = DiabetesPredictionApp()
    app.run()