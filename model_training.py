import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

class DiabetesPredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        
    def initialize_models(self):
        """Initialize multiple high-accuracy models"""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(random_state=42),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=False),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42)
        }
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models and evaluate performance"""
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        return results
    
    def select_best_model(self, results):
        """Select the best model based on ROC-AUC score"""
        best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
        
        return best_model_name, results[best_model_name]
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from the best model"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        return self.feature_importance
    
    def save_model(self, filepath):
        """Save the best model"""
        if self.best_model:
            joblib.dump(self.best_model, filepath)
    
    def load_model(self, filepath):
        """Load a saved model"""
        self.best_model = joblib.load(filepath)
    
    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_model:
            return self.best_model.predict(X), self.best_model.predict_proba(X)
        else:
            raise ValueError("No model has been trained or loaded")