import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_sample_dataset():
    """Create a sample diabetes dataset for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'pregnancies': np.random.randint(0, 17, n_samples),
        'glucose': np.clip(np.random.normal(120, 30, n_samples), 50, 200),
        'blood_pressure': np.clip(np.random.normal(70, 12, n_samples), 40, 120),
        'skin_thickness': np.clip(np.random.normal(20, 10, n_samples), 10, 60),
        'insulin': np.clip(np.random.normal(80, 100, n_samples), 0, 300),
        'bmi': np.clip(np.random.normal(32, 8, n_samples), 15, 50),
        'diabetes_pedigree': np.clip(np.random.normal(0.5, 0.3, n_samples), 0.08, 2.5),
        'age': np.random.randint(21, 81, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic target variable
    glucose_effect = (df['glucose'] > 140).astype(int) * 0.3
    bmi_effect = (df['bmi'] > 30).astype(int) * 0.2
    age_effect = (df['age'] > 45).astype(int) * 0.2
    pedigree_effect = (df['diabetes_pedigree'] > 0.8).astype(int) * 0.3
    
    diabetes_prob = glucose_effect + bmi_effect + age_effect + pedigree_effect
    df['outcome'] = (diabetes_prob + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    
    return df

def save_sample_data():
    """Save sample data to CSV"""
    df = create_sample_dataset()
    df.to_csv('diabetes.csv', index=False)
    print("Sample dataset saved as 'diabetes.csv'")