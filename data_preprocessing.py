import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the diabetes dataset
        You can replace this with your actual dataset
        """
        # For demonstration, we'll create synthetic data
        # In practice, use: df = pd.read_csv('diabetes.csv')
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'pregnancies': np.random.randint(0, 17, n_samples),
            'glucose': np.random.normal(120, 30, n_samples),
            'blood_pressure': np.random.normal(70, 12, n_samples),
            'skin_thickness': np.random.normal(20, 10, n_samples),
            'insulin': np.random.normal(80, 100, n_samples),
            'bmi': np.random.normal(32, 8, n_samples),
            'diabetes_pedigree': np.random.normal(0.5, 0.3, n_samples),
            'age': np.random.randint(21, 81, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable based on realistic patterns
        glucose_effect = (df['glucose'] > 140).astype(int) * 0.3
        bmi_effect = (df['bmi'] > 30).astype(int) * 0.2
        age_effect = (df['age'] > 45).astype(int) * 0.2
        pedigree_effect = (df['diabetes_pedigree'] > 0.8).astype(int) * 0.3
        
        diabetes_prob = glucose_effect + bmi_effect + age_effect + pedigree_effect
        df['outcome'] = (diabetes_prob + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
        
        return df
    
    def preprocess_features(self, df):
        """Preprocess features for training"""
        # Handle missing values (if any)
        df = df.fillna(df.median())
        
        # Separate features and target
        X = df.drop('outcome', axis=1)
        y = df['outcome']
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    def fit_scaler(self, X_train):
        """Fit scaler on training data"""
        self.scaler.fit(X_train)
        self.imputer.fit(X_train)
    
    def transform_features(self, X):
        """Transform features using fitted scaler"""
        X_imputed = self.imputer.transform(X)
        return self.scaler.transform(X_imputed)
    
    def save_preprocessor(self, filepath):
        """Save preprocessor for later use"""
        joblib.dump({
            'scaler': self.scaler,
            'imputer': self.imputer
        }, filepath)
    
    def load_preprocessor(self, filepath):
        """Load preprocessor"""
        preprocessor = joblib.load(filepath)
        self.scaler = preprocessor['scaler']
        self.imputer = preprocessor['imputer']