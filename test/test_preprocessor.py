import sys
import os
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.automl.automl import AutoMLRunner

# Instantiate the runner once for all tests
runner = AutoMLRunner()

# --- Test Fixtures (Optional but good practice for complex setup) ---
@pytest.fixture
def sample_data_cls():
    """ Provides a sample DataFrame for classification preprocessing tests. """
    return pd.DataFrame({
        'numeric_low_skew_nan': [10, 11, np.nan, 12, 10.5, 11.5], # Mean=11, Skew=low -> impute 11
        'numeric_high_skew_nan': [1, 2, np.nan, 100, 3, 5], # Median=3, Skew=high -> impute 3
        'cat_low_card_nan': ['A', 'B', 'A', np.nan, 'A', 'B'], # Mode=A -> impute 'A' -> OHE
        'cat_high_card_nan': ['X', 'Y', 'X', 'Z', 'Y', np.nan], # Mode=X/Y, freq=0.4 < 0.5 -> impute 'Unknown' -> OHE
        'cat_high_missing': ['P', np.nan, np.nan, np.nan, np.nan, 'Q'], # 66% missing <= 0.7 -> keep -> impute 'Unknown' -> OHE
        'cat_single_val_nan': ['S', 'S', np.nan, 'S', 'S', 'S'], # Mode=S -> impute S -> drop in encoding
        'feature_no_nan': [1, 2, 3, 4, 5, 6],
        'target_cls': [0, 1, 0, 1, 0, 1] # Target column
    })

@pytest.fixture
def sample_data_reg(sample_data_cls): # Inject sample_data_cls as an argument
    """ Provides a sample DataFrame for regression preprocessing tests. """
    df = sample_data_cls.copy() # Use the injected fixture
    df['target_reg'] = [1.1, 2.2, 1.5, 3.8, 2.0, 4.5]
    df = df.drop(columns=['target_cls'])
    return df

# --- Test Cases for preprocess_data ---

def test_preprocess_classification_flow(sample_data_cls):
    """ Verify full flow: NaN imputation -> Encoding -> Target Rejoin (Classification)."""
    df_in = sample_data_cls
    target_col = 'target_cls'

    processed_df = runner.preprocess_data(
        df_in.copy(),
        target=target_col,
        task_type='classification',
        threshold_high_missing=0.7, # Keep threshold high to prevent dropping cat_high_missing
        threshold_abs_skewness=1.0,
        cardinality_threshold=5
    )

    # Expected state *after* all preprocessing
    expected_df = pd.DataFrame({
        # Imputed numericals
        'numeric_low_skew_nan': [10.0, 11.0, 11.0, 12.0, 10.5, 11.5], # Mean was 11.0
        'numeric_high_skew_nan': [1.0, 2.0, 3.0, 100.0, 3.0, 5.0], # Median was 3.0
        # OHE for cat_low_card_nan (imputed: A, B, A, A, A, B)
        'cat_low_card_nan_A': [True, False, True, True, True, False],
        'cat_low_card_nan_B': [False, True, False, False, False, True],
        # OHE for cat_high_card_nan (imputed: X, Y, X, Z, Y, 'Unknown')
        'cat_high_card_nan_Unknown': [False, False, False, False, False, True],
        'cat_high_card_nan_X': [True, False, True, False, False, False],
        'cat_high_card_nan_Y': [False, True, False, False, True, False],
        'cat_high_card_nan_Z': [False, False, False, True, False, False],
        # OHE for cat_high_missing (imputed: P, 'Unknown', 'Unknown', 'Unknown', 'Unknown', Q)
        'cat_high_missing_P': [True, False, False, False, False, False],
        'cat_high_missing_Q': [False, False, False, False, False, True],
        'cat_high_missing_Unknown': [False, True, True, True, True, False],
        # cat_single_val_nan was imputed then dropped during encoding
        'feature_no_nan': [1, 2, 3, 4, 5, 6],
        'target_cls': [0, 1, 0, 1, 0, 1] # Original target
    })
    # Ensure dtypes match, especially for boolean columns
    expected_df = expected_df.astype({
        'cat_low_card_nan_A': bool, 'cat_low_card_nan_B': bool,
        'cat_high_card_nan_Unknown': bool, 'cat_high_card_nan_X': bool,
        'cat_high_card_nan_Y': bool, 'cat_high_card_nan_Z': bool,
        'cat_high_missing_P': bool, 'cat_high_missing_Q': bool,
        'cat_high_missing_Unknown': bool
        })


    # Sort columns for comparison consistency
    processed_df = processed_df.sort_index(axis=1)
    expected_df = expected_df.sort_index(axis=1)

    assert_frame_equal(processed_df, expected_df, check_dtype=True) # Check dtypes as well
    assert target_col in processed_df.columns
    assert 'cat_single_val_nan' not in processed_df.columns
    # Check that original categorical columns (except dropped single val) are gone
    assert 'cat_low_card_nan' not in processed_df.columns
    assert 'cat_high_card_nan' not in processed_df.columns
    assert 'cat_high_missing' not in processed_df.columns


def test_preprocess_regression_flow(sample_data_reg):
    """ Verify full flow: NaN imputation -> Encoding -> Target Rejoin (Regression)."""
    df_in = sample_data_reg
    target_col = 'target_reg'

    processed_df = runner.preprocess_data(
        df_in.copy(),
        target=target_col,
        task_type='regression', # Note: task_type doesn't affect preprocess_data logic itself
        threshold_high_missing=0.7, # Keep threshold high to prevent dropping cat_high_missing
        threshold_abs_skewness=1.0,
        cardinality_threshold=5
    )

    # Expected state *after* all preprocessing (features same as classification)
    expected_df = pd.DataFrame({
        'numeric_low_skew_nan': [10.0, 11.0, 11.0, 12.0, 10.5, 11.5],
        'numeric_high_skew_nan': [1.0, 2.0, 3.0, 100.0, 3.0, 5.0],
        'cat_low_card_nan_A': [True, False, True, True, True, False],
        'cat_low_card_nan_B': [False, True, False, False, False, True],
        'cat_high_card_nan_Unknown': [False, False, False, False, False, True],
        'cat_high_card_nan_X': [True, False, True, False, False, False],
        'cat_high_card_nan_Y': [False, True, False, False, True, False],
        'cat_high_card_nan_Z': [False, False, False, True, False, False],
        'cat_high_missing_P': [True, False, False, False, False, False],
        'cat_high_missing_Q': [False, False, False, False, False, True],
        'cat_high_missing_Unknown': [False, True, True, True, True, False],
        'feature_no_nan': [1, 2, 3, 4, 5, 6],
        'target_reg': [1.1, 2.2, 1.5, 3.8, 2.0, 4.5] # Original target
    })
    expected_df = expected_df.astype({
        'cat_low_card_nan_A': bool, 'cat_low_card_nan_B': bool,
        'cat_high_card_nan_Unknown': bool, 'cat_high_card_nan_X': bool,
        'cat_high_card_nan_Y': bool, 'cat_high_card_nan_Z': bool,
        'cat_high_missing_P': bool, 'cat_high_missing_Q': bool,
        'cat_high_missing_Unknown': bool
        })


    processed_df = processed_df.sort_index(axis=1)
    expected_df = expected_df.sort_index(axis=1)

    assert_frame_equal(processed_df, expected_df, check_dtype=True)
    assert target_col in processed_df.columns
    assert 'cat_single_val_nan' not in processed_df.columns
    # Check that original categorical columns (except dropped single val) are gone
    assert 'cat_low_card_nan' not in processed_df.columns
    assert 'cat_high_card_nan' not in processed_df.columns
    assert 'cat_high_missing' not in processed_df.columns

def test_preprocess_target_not_found():
    """ Verify ValueError is raised if target column doesn't exist."""
    df_in = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    with pytest.raises(ValueError, match="Target column 'C' not found"):
        runner.preprocess_data(df_in, target='C', task_type='classification')

def test_preprocess_no_categorical_features():
    """ Verify processing works correctly when no categorical features exist."""
    df_in = pd.DataFrame({
        'num1_nan': [1, 2, np.nan, 4], # Mean = 7/3 = 2.33
        'num2': [5, 6, 7, 8],
        'target': [0, 1, 0, 1]
    })
    expected_df = pd.DataFrame({
        'num1_nan': [1.0, 2.0, 7/3, 4.0],
        'num2': [5, 6, 7, 8],
        'target': [0, 1, 0, 1]
    })

    processed_df = runner.preprocess_data(df_in.copy(), target='target', task_type='classification')

    processed_df = processed_df.sort_index(axis=1)
    expected_df = expected_df.sort_index(axis=1)

    assert_frame_equal(processed_df, expected_df)

def test_preprocess_empty_dataframe():
    """ Verify processing handles empty DataFrame."""
    df_in = pd.DataFrame({'A': [], 'target': []})
    # Expect an empty DF with only the target column
    expected_df = pd.DataFrame({'A': [], 'target': []}) # Match expected output dtype

    processed_df = runner.preprocess_data(df_in.copy(), target='target', task_type='classification')
    processed_df = processed_df.sort_index(axis=1)
    expected_df = expected_df.sort_index(axis=1)
    # Empty dataframes can have different dtypes, check basic structure
    assert processed_df.empty
    assert list(processed_df.columns) == list(expected_df.columns)
    assert processed_df.index.equals(expected_df.index)


    # Case: Empty DF, no target column specified (let's assume target='A')
    df_in_no_target = pd.DataFrame({'A': []})
    with pytest.raises(ValueError, match="Target column 'target' not found"):
        runner.preprocess_data(df_in_no_target.copy(), target='target', task_type='classification')

    # Case: Completely empty DF
    df_in_empty = pd.DataFrame()
    with pytest.raises(ValueError, match="Target column 'A' not found"):
         runner.preprocess_data(df_in_empty.copy(), target='A', task_type='classification')


def test_preprocess_only_target_column():
    """ Verify processing handles DataFrame with only the target column."""
    df_in = pd.DataFrame({'target': [1, 2, 3]})
    # Expect a DF with only the target column
    expected_df = df_in.copy()
    processed_df = runner.preprocess_data(df_in.copy(), target='target', task_type='regression')
    processed_df = processed_df.sort_index(axis=1)
    expected_df = expected_df.sort_index(axis=1)
    assert_frame_equal(processed_df, expected_df)