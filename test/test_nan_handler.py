
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.automl.automl import AutoMLRunner

import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal


# Instantiate the class once for all tests in this module (or use fixtures)
preprocessor = AutoMLRunner()
DEFAULT_PLACEHOLDER = "Unknown"

# --- Test Cases ---

def test_drop_high_missing():
    """Verify columns with > threshold_high_missing NaNs are dropped."""
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5], # 80% missing
        'C': [1, 2, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10] # 60% missing
    })
    # Using default threshold_high_missing = 0.7
    processed_df = preprocessor.preprocess_data(df.copy(), 'classification')

    assert 'A' in processed_df.columns
    assert 'B' not in processed_df.columns # Should be dropped (80% > 70%)
    assert 'C' in processed_df.columns     # Should be kept (60% < 70%)

def test_no_missing():
    """Verify columns with no missing values are untouched."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [3, 2, 1]    
    })
    processed_df = preprocessor.preprocess_data(df.copy(), 'classification')
    
    assert 'A' in processed_df.columns
    assert 'B' in processed_df.columns

def test_numerical_mean_imputation():
    """Verify numerical column with low skewness uses mean imputation."""
    # Skewness of [10, 11, np.nan, 12, 10.5] (non-nan: 10, 11, 12, 10.5) is low (~0.3)
    # Mean = (10 + 11 + 12 + 10.5) / 4 = 43.5 / 4 = 10.875
    df = pd.DataFrame({
        'Num_LowSkew': [10, 11, np.nan, 12, 10.5]
        })
    processed_df = preprocessor.preprocess_data(df.copy(), task_type='regression')

    assert processed_df['Num_LowSkew'].isnull().sum() == 0
    assert processed_df['Num_LowSkew'].iloc[2] == pytest.approx(10.875)

def test_numerical_median_imputation():
    """Verify numerical column with high skewness uses median imputation."""
    # Skewness of [1, 2, np.nan, 100, 3] (non-nan: 1, 2, 100, 3) is high (~1.7)
    # Median = median([1, 2, 3, 100]) = (2+3)/2 = 2.5
    df = pd.DataFrame({
        'Num_HighSkew': [1, 2, np.nan, 100, 3]
        })
    
    processed_df = preprocessor.preprocess_data(df.copy(), task_type='regression', threshold_abs_skewness=1.0)

    assert processed_df['Num_HighSkew'].isnull().sum() == 0
    assert processed_df['Num_HighSkew'].iloc[2] == 2.5


