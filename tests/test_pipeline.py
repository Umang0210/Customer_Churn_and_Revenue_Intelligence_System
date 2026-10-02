import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import main as build_features

def test_feature_engineering():
    # Simple placeholder to ensure imports and environments are correct
    # In a real test, we would mock the database connection or use a test DB
    pass

def test_missing_values_handled():
    # Example logic test for data cleaning
    df = pd.DataFrame({
        'TotalCharges': ['10.5', ' ', '20.0'],
        'tenure': [1, 0, 5]
    })
    
    # Simulate cleaning logic
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    assert pd.isna(df['TotalCharges'].iloc[1])
    
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    assert df['TotalCharges'].iloc[1] == 0.0
