# tests/test_data_processing.py - UNIT TESTS
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestDataProcessing(unittest.TestCase):
    
    def test_data_loading(self):
        """Test that data files exist and can be loaded"""
        # Test 1: Raw data exists
        self.assertTrue(os.path.exists('data/raw/train.csv'), 
                       "train.csv should exist in data/raw/")
        
        # Test 2: Can load raw data
        df = pd.read_csv('data/raw/train.csv')
        self.assertGreater(len(df), 0, "Data should have rows")
        self.assertIn('CustomerId', df.columns, "Should have CustomerId column")
        self.assertIn('Amount', df.columns, "Should have Amount column")
    
    def test_rfm_features(self):
        """Test that RFM features were created correctly"""
        # Test 1: Customer features file exists
        self.assertTrue(os.path.exists('data/processed/customer_features.csv'),
                       "customer_features.csv should exist")
        
        # Test 2: Has required columns
        rfm = pd.read_csv('data/processed/customer_features.csv')
        required_cols = ['CustomerId', 'recency', 'frequency', 'total_amount']
        for col in required_cols:
            self.assertIn(col, rfm.columns, f"Should have {col} column")
    
    def test_target_variable(self):
        """Test that target variable was created"""
        # Test 1: Target file exists
        self.assertTrue(os.path.exists('data/processed/target_variable.csv'),
                       "target_variable.csv should exist")
        
        # Test 2: Has binary target
        target = pd.read_csv('data/processed/target_variable.csv')
        self.assertIn('is_high_risk', target.columns, 
                     "Should have is_high_risk column")
        
        # Test 3: Target values are 0 or 1
        unique_values = target['is_high_risk'].unique()
        self.assertTrue(set(unique_values).issubset({0, 1}),
                       "is_high_risk should only have values 0 or 1")
    
    def test_feature_ranges(self):
        """Test that features have reasonable ranges"""
        rfm = pd.read_csv('data/processed/customer_features.csv')
        
        # Test recency is positive
        self.assertTrue((rfm['recency'] >= 0).all(), 
                       "Recency should be non-negative")
        
        # Test frequency is positive
        self.assertTrue((rfm['frequency'] > 0).all(), 
                       "Frequency should be positive")
        
        # Test total_amount is reasonable (not extremely large)
        self.assertTrue((rfm['total_amount'].abs() < 1e9).all(),
                       "Total amount should be reasonable")

if __name__ == '__main__':
    unittest.main()