import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.automl.automl import AutoMLRunner

import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

runner = AutoMLRunner()

def test_encode_low_cardinality_ohe():
    """Verify low cardinality column uses One-Hot Encoding."""
    df_in = pd.DataFrame({
        'cat_low': ['A', 'B', 'A', 'C', 'B'],
        'other': [1, 2, 3, 4, 5]
    })
    
    expected_df = pd.DataFrame({
        'other': [1, 2, 3, 4, 5],
        'cat_low_A': [True, False, True, False, False],
        'cat_low_B': [False, True, False, False, True],
        'cat_low_C': [False, False, False, True, False]
    })
    
    expected_df = expected_df.astype({
        'cat_low_A': 'bool',
        'cat_low_B': 'bool',
        'cat_low_C': 'bool'
    })

    processed_df = runner._encode_categorical_features(df_in.copy(), 'cat_low', cardinality_threshold=5)
    
    # Sort columns for consistent comparison
    processed_df = processed_df.sort_index(axis=1)
    expected_df = expected_df.sort_index(axis=1)
    
    assert_frame_equal(processed_df, expected_df)
    
def test_encode_high_cardinality_frequency():
    """Verify high cardinality column uses Frequency Encoding."""
    df_in = pd.DataFrame({
        'cat_high': ['X', 'Y', 'X', 'Z', 'Y', 'X', 'W', 'V', 'U', 'T'],
        'numeric': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    })
    
    # Frequencies: X=0.3, Y=0.2, Z=0.1, W=0.1, V=0.1, U=0.1, T=0.1
    expected_df = pd.DataFrame({
        'numeric': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'cat_high_freq': [0.3, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1]
    })

    processed_df = runner._encode_categorical_features(df_in.copy(), 'cat_high', cardinality_threshold=5)

    # Sort columns for consistent comparison
    processed_df = processed_df.sort_index(axis=1)
    expected_df = expected_df.sort_index(axis=1)

    assert_frame_equal(processed_df, expected_df)
    
def test_encode_single_value_drops():
    """Verify column with only one unique value is dropped."""
    df_in = pd.DataFrame({
        'single': ['K', 'K', 'K', 'K'],
        'feature': [1, 1, 2, 2]
    })
    expected_df = pd.DataFrame({
        'feature': [1, 1, 2, 2]
    })
    processed_df = runner._encode_categorical_features(df_in.copy(), 'single')
    assert_frame_equal(processed_df, expected_df)
    
def test_encode_with_nan_values():
    """Verify encoding works if NaNs are present (though ideally handled before)."""
    # Note: NaNs should ideally be imputed *before* encoding.
    # pd.get_dummies(dummy_na=False) ignores NaNs.
    # Frequency encoding will calculate frequency excluding NaNs.
    df_in_ohe = pd.DataFrame({
        'cat_low_nan': ['A', 'B', np.nan, 'A', 'C'],
        'val': [1,2,3,4,5]
        })
    expected_ohe = pd.DataFrame({
        'val': [1, 2, 3, 4, 5],
        'cat_low_nan_A': [True, False, False, True, False],
        'cat_low_nan_B': [False, True, False, False, False],
        'cat_low_nan_C': [False, False, False, False, True]
    })
    expected_ohe = expected_ohe.astype({'cat_low_nan_A': 'bool', 'cat_low_nan_B': 'bool', 'cat_low_nan_C': 'bool'})

    processed_ohe = runner._encode_categorical_features(df_in_ohe.copy(), 'cat_low_nan', cardinality_threshold=5)
    processed_ohe = processed_ohe.sort_index(axis=1)
    expected_ohe = expected_ohe.sort_index(axis=1)
    assert_frame_equal(processed_ohe, expected_ohe)

    # --- Frequency encoding with NaN ---
    df_in_freq = pd.DataFrame({
        'cat_high_nan': ['X', 'Y', 'X', np.nan, 'Y', 'X'],
        'val': [1,2,3,4,5,6]
        })
    # Frequencies based on non-NaN: X=3/5=0.6, Y=2/5=0.4
    expected_freq = [0.6, 0.4, 0.6, np.nan, 0.4, 0.6] # NaN remains NaN after mapping
    expected_freq_df = pd.DataFrame({
        'val': [1, 2, 3, 4, 5, 6],
        'cat_high_nan_freq': expected_freq
    })

    processed_freq = runner._encode_categorical_features(df_in_freq.copy(), 'cat_high_nan', cardinality_threshold=1)
    processed_freq = processed_freq.sort_index(axis=1)
    expected_freq_df = expected_freq_df.sort_index(axis=1)
    assert_frame_equal(processed_freq, expected_freq_df)
    
def test_encode_non_object_column_skipped():
    """Verify non-object/category columns are skipped."""
    df_in = pd.DataFrame({
        'numeric': [1, 2, 3, 4],
        'string': ['a', 'b', 'a', 'c']
    })
    # Expect only 'string' to be encoded, 'numeric' remains untouched
    expected_df = pd.DataFrame({
        'numeric': [1, 2, 3, 4],
        'string_a': [True, False, True, False],
        'string_b': [False, True, False, False],
        'string_c': [False, False, False, True]
    })
    expected_df = expected_df.astype({'string_a': 'bool', 'string_b': 'bool', 'string_c': 'bool'})

    # Try encoding the numeric column - should return original df
    processed_df_no_change = runner._encode_categorical_features(df_in.copy(), 'numeric')
    assert_frame_equal(processed_df_no_change, df_in) # No change expected

    # Encode the string column - should work
    processed_df_encoded = runner._encode_categorical_features(df_in.copy(), 'string', cardinality_threshold=5)
    processed_df_encoded = processed_df_encoded.sort_index(axis=1)
    expected_df = expected_df.sort_index(axis=1)
    assert_frame_equal(processed_df_encoded, expected_df)
    
def test_encode_column_not_found():
    """Verify function handles non-existent column name gracefully."""
    df_in = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
    processed_df = runner._encode_categorical_features(df_in.copy(), 'C') # Column 'C' doesn't exist
    # Should return the original DataFrame without error
    assert_frame_equal(processed_df, df_in)
    
def test_encode_empty_dataframe():
    """Verify function handles empty DataFrame."""
    df_in = pd.DataFrame({'A': []}) # Empty df with column
    processed_df = runner._encode_categorical_features(df_in.copy(), 'A')
    assert_frame_equal(processed_df, df_in) # Expect empty back

    df_in_no_cols = pd.DataFrame() # Empty df no columns
    processed_df_no_cols = runner._encode_categorical_features(df_in_no_cols.copy(), 'A')
    assert_frame_equal(processed_df_no_cols, df_in_no_cols) # Expect empty back