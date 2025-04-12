import pandas as pd
from sklearn.datasets import make_regression
import numpy as np

def generate_regression_dataset(
    n_samples: int = 100,
    n_features: int = 2,
    n_informative: int = 2,
    noise: float = 0.1,
    random_state: int = 42,
    bias: float = 0.0,
) -> pd.DataFrame:
    """
    Generates a synthetic dataset for regression tasks using scikit-learn's make_regression.

    Args:
        n_samples (int, optional): The number of samples. Defaults to 100.
        n_features (int, optional): The number of features. Defaults to 2.
        n_informative (int, optional): The number of informative features. Defaults to 2.
        noise (float, optional): The standard deviation of the Gaussian noise applied to the output. Defaults to 0.1.
        random_state (int, optional): The seed for the random number generator. Defaults to 42.
        bias (float, optional): Bias term in the underlying linear model. Defaults to 0.0

    Returns:
        pd.DataFrame: A DataFrame containing the generated dataset.
            The features are named 'feature_0', 'feature_1', ..., 'feature_{n_features-1}'.
            The target variable is named 'target'.
    """
    # Generate the regression dataset
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state,
        bias=bias,
    )

    # Convert to pandas DataFrame
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    y = pd.Series(y, name="target")
    df = pd.concat([X, y], axis=1)
    return df

if __name__ == "__main__":
    # Generate a sample dataset
    regression_data = generate_regression_dataset(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        noise=10,
        random_state=100,
        bias= 5
    )

    # Print the first 5 rows
    print(regression_data.head())

    # Print the shape of the DataFrame
    print(f"Shape of the dataset: {regression_data.shape}")

    # Display info about the DataFrame
    regression_data.info()

    # You can save it to a CSV file
    regression_data.to_csv("regression_dataset.csv", index=False)
    print("Dataset saved to regression_dataset.csv")