import joblib
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


# =====================
# CUSTOM TRANSFORMERS
# =====================

class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._feature_names_in = None

    def fit(self, X, y=None):
        # Store feature names using scikit-learn's built-in validation
        self._check_feature_names(X, reset=True)
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Combine thinness features
        X['thinness'] = X[['thinness1-19', 'thinness5-9']].mean(axis=1)
        X.drop(columns=['thinness1-19', 'thinness5-9'], inplace=True)
        
        # 2. Handle Income zeros
        X['Income'] = X['Income'].replace(0, np.nan)
        
        return X

class DynamicKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.knn = None
        
    def fit(self, X, y=None):
        # Calculate k based on number of samples in training data
        k = int(np.sqrt(X.shape[0]))
        self.knn = KNNImputer(n_neighbors=k)
        self.knn.fit(X)
        return self
    
    def transform(self, X):
        return self.knn.transform(X)


# =====================
# MODEL PREDICTOR
# =====================

class LifeExpectancyPredictor:
    def __init__(self):
        self.model = self._load_model()
        
    def _load_model(self):
        try:
            return joblib.load("./life_expectancy.joblib")
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

    def predict(self, input_data: dict) -> float:
        """Process input through full pipeline"""
        input_df = pd.DataFrame([input_data], columns=self.model.feature_names_in_)
        return self.model.predict(input_df)[0]

    def _validate_input(self, input_data: dict):
        """Validate input features match training features"""
        expected_features = set(self.model.feature_names_in_)
        provided_features = set(input_data.keys())
        
        if missing := expected_features - provided_features:
            raise ValueError(f"Missing features: {missing}")
            
        if extra := provided_features - expected_features:
            raise ValueError(f"Unexpected features: {extra}")
            
        return input_data


# =====================
# SIMPLE GUI
# =====================

FIELDS = [
    "AdultMortality", "Alcohol", "percentExpenditure",
    "Polio", "Total expenditure", "Diphtheria",
    "HIV/AIDS", "GDP", "thinness1-19", "thinness5-9",
    "Income", "Schooling"  # Removed Measles, HepatitisB, Population, InfantDeaths
]

# Default values for each field in order
DEFAULT_VALUES = [
    144.0, 3.755, 64.91291, 93.0, 5.755,
    93.0, 0.1, 1766.948, 3.3, 3.3, 0.677, 12.3
]
        
class LifeExpectancyApp:
    def __init__(self, master):
        self.master = master
        self.predictor = LifeExpectancyPredictor()
        self._create_widgets()
        
    def _create_widgets(self):
        """Create all GUI components"""
        self.entries = {}

        
        # Create labels and entries for each field
        for idx, field in enumerate(FIELDS):
            label = tk.Label(self.master, text=field)
            label.grid(row=idx, column=0)
            entry = tk.Entry(self.master)
            entry.grid(row=idx, column=1)
            # Insert default value
            entry.insert(0, str(DEFAULT_VALUES[idx]))
            self.entries[field] = entry
            
        # Add buttons and result label
        self.btn_predict = tk.Button(
            self.master, 
            text="Predict", 
            command=self._handle_prediction
        )
        self.btn_predict.grid(row=len(FIELDS)+1, columnspan=2)
        
        self.lbl_result = tk.Label(self.master, text="")
        self.lbl_result.grid(row=len(FIELDS)+2, columnspan=2)
        

    def _handle_prediction(self):
        """Handle prediction button click"""
        try:
            input_data = {k: float(v.get()) for k, v in self.entries.items()}
            # Directly use the predictor's predict method
            prediction = self.predictor.predict(input_data)
            self.lbl_result.config(text=f"Life Expectancy: {prediction:.2f} years")
        except Exception as e:
            messagebox.showerror("Error", str(e))


# =====================
# RUN APPLICATION
# =====================

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Life Expectancy Predictor")
    app = LifeExpectancyApp(root)
    root.mainloop()