# test/test_automl_pipeline.py

import sys
import os
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.automl.automl import AutoMLRunner

# --- Reusable Fixtures ---

@pytest.fixture(scope="module")
def runner_instance():
    """Provides a single AutoMLRunner instance for the test module."""
    return AutoMLRunner()

@pytest.fixture
def sample_data_cls():
    """ Provides a sample DataFrame for classification pipeline tests. """
    # Simplified version sufficient for pipeline flow testing
    return pd.DataFrame({
        'numeric_feat': [10, 11, 11.5, 12, 10.5, 11.5, 9, 13],
        'cat_feat': ['A', 'B', 'A', 'B', 'A', 'A', 'B', 'A'],
        'target_cls': [0, 1, 0, 1, 0, 0, 1, 0]
    })

@pytest.fixture
def sample_data_reg():
    """ Provides a sample DataFrame for regression pipeline tests. """
    return pd.DataFrame({
        'numeric_feat_1': [1, 2, 3, 4, 5, 6, 7, 8],
        'numeric_feat_2': [10, 12, 11, 13, 14, 12, 15, 11],
        'target_reg': [1.1, 2.2, 1.5, 3.8, 2.0, 4.5, 3.0, 4.0]
    })

@pytest.fixture
def sample_data_with_nans():
    """ Provides data with NaNs to test preprocessing integration. """
    return pd.DataFrame({
        'numeric_low_skew_nan': [10, 11, np.nan, 12, 10.5, 11.5], # Mean=11
        'cat_low_card_nan': ['A', 'B', 'A', np.nan, 'A', 'B'], # Mode=A
        'target_cls': [0, 1, 0, 1, 0, 1]
    })

@pytest.fixture
def sample_data_preprocess_failure():
    """ Data where preprocessing might drop all features. """
    return pd.DataFrame({
        'feat_high_nan': [np.nan, np.nan, np.nan, 1, np.nan, np.nan], # >70% NaN -> dropped
        'target_cls': [0, 1, 0, 1, 0, 1]
    })

@pytest.fixture
def sample_data_split_failure():
    """ Data too small for standard splitting. """
    return pd.DataFrame({
        'feature': [1, 2, 3],
        'target_cls': [0, 1, 0]
    })

# --- Tests for _split_data ---

def test_split_data_classification(runner_instance, sample_data_cls):
    X = sample_data_cls.drop('target_cls', axis=1)
    y = sample_data_cls['target_cls']
    # Preprocess slightly just to make it numeric for splitting
    X = pd.get_dummies(X, columns=['cat_feat'], drop_first=True)

    split_result = runner_instance._split_data(X, y, 'classification')

    assert split_result is not None
    assert len(split_result) == 4
    X_train, X_test, y_train, y_test = split_result
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    # Default test_size=0.2 -> 8 * 0.8 = 6.4 -> 6 train, 2 test (approx)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert 1 <= len(X_test) <= 3

def test_split_data_regression(runner_instance, sample_data_reg):
    X = sample_data_reg.drop('target_reg', axis=1)
    y = sample_data_reg['target_reg']

    split_result = runner_instance._split_data(X, y, 'regression')

    assert split_result is not None
    assert len(split_result) == 4

    X_train, X_test, y_train, y_test = split_result
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)

def test_split_data_insufficient_data_fallback(runner_instance, sample_data_split_failure):
    """ Test when stratification fails, it falls back to regular split """
    X = sample_data_split_failure[['feature']]
    y = sample_data_split_failure['target_cls']

    split_result = runner_instance._split_data(X, y, 'classification')
    
    assert split_result is not None
    assert len(split_result) == 4
    X_train, X_test, y_train, y_test = split_result
    assert len(X_train) > 0
    assert len(X_test) > 0

# --- Tests for _select_and_train_default_model ---

def test_select_train_classification(runner_instance):
    X_train = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    model = runner_instance._select_and_train_default_model(X_train, y_train, 'classification')
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_") # Check if model appears fitted

def test_select_train_regression(runner_instance):
    X_train = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6]})
    y_train = pd.Series([1.1, 2.2, 3.3])
    model = runner_instance._select_and_train_default_model(X_train, y_train, 'regression')
    assert isinstance(model, LinearRegression)
    assert hasattr(model, "coef_") # Check if model appears fitted

def test_select_train_invalid_task(runner_instance):
    X_train = pd.DataFrame({'f1': [1, 2, 3]})
    y_train = pd.Series([0, 1, 0])
    model = runner_instance._select_and_train_default_model(X_train, y_train, 'invalid_task')
    assert model is None

# --- Tests for _evaluate_model --- (Using capsys to check print output)

def test_evaluate_classification(runner_instance, capsys):
    X_test = pd.DataFrame({'f1': [1, 2], 'f2': [4, 5]})
    y_test = pd.Series([0, 1])
    # Dummy fitted model
    model = LogisticRegression().fit(X_test, y_test)

    runner_instance._evaluate_model(model, X_test, y_test, 'classification')
    captured = capsys.readouterr()
    assert "**_evaluate_model**: Evaluation Results (Classification):" in captured.out
    assert "Model Accuracy:" in captured.out
    assert "Classification Report:" in captured.out

def test_evaluate_regression(runner_instance, capsys):
    X_test = pd.DataFrame({'f1': [1, 2], 'f2': [4, 5]})
    y_test = pd.Series([1.1, 2.2])
    # Dummy fitted model
    model = LinearRegression().fit(X_test, y_test)

    runner_instance._evaluate_model(model, X_test, y_test, 'regression')
    captured = capsys.readouterr()
    assert "**_evaluate_model**: Evaluation Results (Regression):" in captured.out
    assert "Mean Squared Error:" in captured.out
    assert "R-squared:" in captured.out

# --- Tests for _prepare_data_for_modeling ---

def test_prepare_data_classification_success(runner_instance, sample_data_cls):
    result = runner_instance._prepare_data_for_modeling(sample_data_cls.copy(), 'target_cls')
    assert result is not None
    assert len(result) == 5
    X_train, X_test, y_train, y_test, task_type = result
    assert task_type == 'classification'
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    assert 'cat_feat' not in X_train.columns
    assert 'cat_feat_B' in X_train.columns

def test_prepare_data_regression_success(runner_instance, sample_data_reg):
    result = runner_instance._prepare_data_for_modeling(sample_data_reg.copy(), 'target_reg')
    assert result is not None
    assert len(result) == 5
    _, _, _, _, task_type = result
    assert task_type == 'regression'

def test_prepare_data_with_nans_success(runner_instance, sample_data_with_nans):
    """ Verify it handles NaNs during preparation """
    result = runner_instance._prepare_data_for_modeling(sample_data_with_nans.copy(), 'target_cls')
    assert result is not None
    assert len(result) == 5
    X_train, _, _, _, task_type = result
    assert task_type == 'classification'

    assert not X_train['numeric_low_skew_nan'].isnull().any()
    assert 'cat_low_card_nan_B' in X_train.columns

def test_prepare_data_target_missing(runner_instance, sample_data_cls):
    result = runner_instance._prepare_data_for_modeling(sample_data_cls.copy(), 'non_existent_target')
    assert result is None

def test_prepare_data_preprocessing_failure_scenario(runner_instance, sample_data_preprocess_failure):
    """ Test scenario where preprocessing removes all features """
    result = runner_instance._prepare_data_for_modeling(sample_data_preprocess_failure.copy(), 'target_cls')
    assert result is None

def test_prepare_data_split_fallback_scenario(runner_instance, sample_data_split_failure): # Renamed test
    """ Test scenario where splitting falls back successfully within prepare_data """
    result = runner_instance._prepare_data_for_modeling(sample_data_split_failure.copy(), 'target_cls')

    assert result is not None 
    assert len(result) == 5
    X_train, X_test, y_train, y_test, task_type = result
    assert task_type == 'classification' 


# --- Tests for run_baseline_pipeline (Integration) ---

def test_run_pipeline_classification_success(runner_instance, sample_data_cls):
    target = 'target_cls'
    df = sample_data_cls.copy()
    model = runner_instance.run_baseline_pipeline(df, target)

    assert model is not None
    assert isinstance(model, LogisticRegression)
    assert runner_instance.model == model
    assert hasattr(model, "predict")

def test_run_pipeline_regression_success(runner_instance, sample_data_reg):
    target = 'target_reg'
    df = sample_data_reg.copy()
    model = runner_instance.run_baseline_pipeline(df, target)

    assert model is not None
    assert isinstance(model, LinearRegression)
    assert runner_instance.model == model
    assert hasattr(model, "predict")

def test_run_pipeline_target_missing(runner_instance, sample_data_cls):
    target = 'non_existent_target'
    df = sample_data_cls.copy()
    model = runner_instance.run_baseline_pipeline(df, target)
    assert model is None
    assert runner_instance.model is None

def test_run_pipeline_prepare_fails(runner_instance, sample_data_preprocess_failure):
    """ Test pipeline end-to-end when _prepare_data returns None """
    target = 'target_cls'
    df = sample_data_preprocess_failure.copy()
    model = runner_instance.run_baseline_pipeline(df, target)
    assert model is None
    assert runner_instance.model is None