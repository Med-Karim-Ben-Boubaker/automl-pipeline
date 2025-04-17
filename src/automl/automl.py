import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split   
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

class AutoMLRunner:
    def __init__(self):
        self.data_frame = None
        self.task_type = None
    
    def load_csv_data(self, file_path: str) -> pd.DataFrame:
        
        try:
            df = pd.read_csv(file_path)
            self.df = df
            return df
        
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        
        except pd.errors.ParserError:
            print(f"Error: Could not parse CSV file at {file_path}. Check file format.")
            return None
        
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
        
    def get_task_type(self, dataframe: pd.DataFrame, target: str) -> str | None:
        """
        Determines the task type (classification or regression) based on the target column's dtype and unique values.
        - Object/Category/Bool -> Classification
        - Integer with few unique values -> Classification
        - Integer with many unique values -> Regression
        - Float -> Regression
        """
        if target not in dataframe.columns:
            print(f"**get_task_type**: Error: Target column '{target}' not found.")
            return None

        target_series = dataframe[target].dropna()

        if target_series.empty:
             print(f"**get_task_type**: Warning: Target column '{target}' has no valid data after dropping NaNs.")
             return None

        dtype = target_series.dtype
        n_unique = target_series.nunique()
        unique_value_threshold = 20

        if pd.api.types.is_object_dtype(dtype) or \
           isinstance(dtype, pd.CategoricalDtype) or \
           pd.api.types.is_bool_dtype(dtype):
            task = 'classification'
        elif pd.api.types.is_integer_dtype(dtype):
            if n_unique <= unique_value_threshold:
                task = 'classification'
            else:
                task = 'regression'
        elif pd.api.types.is_float_dtype(dtype):
             if n_unique <= 2:
                 print(f"**get_task_type**: Warning: Float target '{target}' has <= 2 unique values. Interpreting as classification.")
                 task = 'classification'
             else:
                 task = 'regression'
        else:
            print(f"**get_task_type**: Warning: Unhandled dtype '{dtype}' for target '{target}'. Defaulting to classification.")
            task = 'classification'

        self.task_type = task
        return task
        
    def _impute_categorical_nan(
        self,
        series: pd.Series,
        column_name: str,
        default_placeholder: str = "Unknown",
        threshold_high_cardinality: float = 0.8,
        threshold_low_mode_freq: float = 0.5,
    ) -> pd.Series :
        """
        Impute NaN values in a categorical series.
        """
        fun_name = "_impute_categorical_nan"
        
        try:
            
            col_non_missing = series.dropna()
            n_non_missing = len(col_non_missing)
            
            if n_non_missing == 0:
                print(f"**{fun_name}**:  Warning: All values are NaN in '{column_name}'.\
                    Filling NaN with {default_placeholder}.")
                
                return series.fillna(default_placeholder)
            
            n_unique = col_non_missing.nunique()
            cardinality_ratio = n_unique / n_non_missing
            print(f"**{fun_name}**:   Unique values (non-NaN): {n_unique}, Cardinality Ratio: \
                {cardinality_ratio:.2f}")
            
            use_placeholder = False
            reason = ""
            
            if cardinality_ratio > threshold_high_cardinality:
                use_placeholder = True
                reason = f"High cardinality ratio ({cardinality_ratio:.2f} > \
                    {threshold_high_cardinality:.2f})"
            
            else:
                modes = col_non_missing.mode()
                if modes.empty:
                    use_placeholder = True
                    reason = f"No mode found (all values unique)"
                else:
                    mode_val = modes[0]
                    mode_freq = (col_non_missing == mode_val).sum() / n_non_missing
                    print(f"**{fun_name}**:  Mode: '{mode_val}', Mode Frequency: {mode_freq:.2f}")
                    
                    if mode_freq < threshold_low_mode_freq:
                        use_placeholder = True
                        reason = f"Low mode frequency ({mode_freq:.2f} < {threshold_low_mode_freq:.2f})"
                        
            if use_placeholder:
                print(f"  Reason for placeholder: {reason}")
                print(f"**{fun_name}**:  Action: Filling NaNs with Placeholder ('{default_placeholder}').")
                return series.fillna(default_placeholder)
            else:
                print(f"**{fun_name}**:  Action: Filling NaNs with Mode ('{mode_val}').")
                return series.fillna(mode_val)
            
        except Exception as e:
            print(f"**{fun_name}**:  Warning: Error during categorical imputation for \
                '{column_name}': {e}. Skipping imputation.")
        
        return series
        
    def _impute_numerical_nan(
        self,
        series: pd.Series,
        column_name: str,
        threshold_abs_skewness: float,
        ) -> pd.Series:
        
        """
        Impute NaN values in a numerical series based on skewness.
        """
        fun_name = "_impute_numerical_nan"
        
        try:
            
            skewness = series.dropna().skew()
            
            if pd.isna(skewness):
                print(f"**{fun_name}**:  Skewness: nan (cannot calculate for '{column_name}')")
                mean_val = series.mean()

                if pd.isna(mean_val):
                    print(f"**{fun_name}**:  Warning: Cannot compute mean for  '{column_name}'\
                        (likely all NaNs). Skipping imputation.")
                    
                    return series
                
                else:
                    print(f"**{fun_name}**:  Action: Imputing NaNs with Mean ({mean_val:.2f})\
                        (due to NaN skewness).")
                    return series.fillna(mean_val)
                
            print(f"**{fun_name}**:  Skewness: {skewness:.2f}")
            
            if abs(skewness) > threshold_abs_skewness:
                median_val = series.median()
                print(f"**{fun_name}**:  Action: Imputing NaNs with Median ({median_val:.2f})\
                    due to high skewness.")
                return series.fillna(median_val)
            
            else:
                mean_val = series.mean()
                print(f"**{fun_name}**:  Action: Imputing NaNs with Mean ({mean_val:.2f})\
                    due to low skewness.")
                return series.fillna(mean_val)
            
        except Exception as e:
            print(f"**{fun_name}**:  Warning: Error during numerical imputation for \
                '{column_name}': {e}. Skipping imputation.")
            
            return series
        
    def preprocess_nan_data(
        self,
        dataframe: pd.DataFrame,
        threshold_high_missing: float = 0.7,
        threshold_abs_skewness: float = 1.0,
    ) -> pd.DataFrame:
        
        """
        Preprocess the DataFrame by handling missing values.
        """
        fun_name = "preprocess_nan_data"
        
        print(f"\n**{fun_name}**: --- Starting NaN Handling ---")
        
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"**{fun_name}**: Input 'dataframe' must be a pandas DataFrame.")
        
        if dataframe.empty:
            print(f"**{fun_name}**: Input DataFrame is empty. Returning empty DataFrame.")
            return dataframe
        
        df_processed = dataframe.copy()
        initial_cols = df_processed.shape[1]
        cols_to_process = df_processed.columns.tolist()
        cols_dropped = 0
        
        for column in cols_to_process:
            
            if column not in df_processed.columns:
                continue
            
            series = df_processed[column]
            missing_percentage = series.isnull().mean()
            print(f"**{fun_name}**:  \nProcessing column: '{column}' \
                (Missing: {missing_percentage:.2%})")
            
            # 1. Check for High Missingness
            if missing_percentage > threshold_high_missing:
                print(f"**{fun_name}**:  Action: Dropping column '{column}' due to high \
                    missing \values ({missing_percentage:.2%}).")
                df_processed.drop(column, axis=1, inplace=True)
                cols_dropped += 1
                continue
            
            # 2. Skip of No Missing Values 
            if missing_percentage == 0:
                print(f"**{fun_name}**:  Action: Skipping column '{column}' (no missing values).")
                continue
            
            # 3. Delegate Imputation based on Type
            imputed_series = None
            if pd.api.types.is_numeric_dtype(series):
                imputed_series = self._impute_numerical_nan(
                    series,
                    column,
                    threshold_abs_skewness
                    )
                
            elif pd.api.types.is_object_dtype(series) or \
                pd.api.types.is_bool_dtype(series):
                    imputed_series = self._impute_categorical_nan(
                        series,
                        column
                    )
                    
            else:
                # -- TO-DO --
                # Handle different datatypes(e.g., datetime )
                print(f"**{fun_name}**: Action: Skipping column '{column}' \
                    (unhandled type: {series.dtype}).")
                
                continue
            
            df_processed[column] = imputed_series
            
            final_cols = df_processed.shape[1]
            print(f"\n**{fun_name}**:  --- NaN Handling Finished ---")
            print(f"**{fun_name}**:  Columns processed. Initial: {initial_cols}, \
                Final: {final_cols}. Dropped: {cols_dropped}")
            
        return df_processed
    
    def preprocess_data(
        self,
        dataframe: pd.DataFrame,
        target: str,
        task_type: str,
        threshold_high_missing: float = 0.7,
        threshold_abs_skewness: float = 1.0,
        cardinality_threshold: int = 10
    ) -> pd.DataFrame | None:
        """
        Full preprocessing pipeline including NaN handling and encoding.
        Returns a DataFrame with processed features and the original target column,
        or None if all feature columns are dropped during preprocessing.
        """
        fun_name = "preprocess_data"
        print(f"**{fun_name}**: --- Starting Full Preprocessing ---")

        if target not in dataframe.columns:
             raise ValueError(f"**{fun_name}**: Target column '{target}' not found in DataFrame.")

        # Handle case where only target column exists initially
        if dataframe.shape[1] <= 1 and target in dataframe.columns:
            print(f"**{fun_name}**: DataFrame contains only the target column. No features to preprocess.")
            return None # Return None as there are no features

        X = dataframe.drop(columns=[target])
        y = dataframe[target]

        # 1. Handle NaNs in features
        X_processed = self.preprocess_nan_data(
            X,
            threshold_high_missing=threshold_high_missing,
            threshold_abs_skewness=threshold_abs_skewness
        )

        if X_processed.empty:
            print(f"**{fun_name}**: Warning: Feature DataFrame is empty after NaN handling.")
            return None # Return None if no features remain

        # 2. Encode Categorical Features
        print(f"\n**{fun_name}**: --- Starting Categorical Encoding ---")
        cols_to_encode = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"**{fun_name}**: Found categorical columns in processed features: {cols_to_encode}")

        temp_X_processed = X_processed.copy()
        for col in cols_to_encode:
            if col in temp_X_processed.columns: # Check if column still exists
                 temp_X_processed = self._encode_categorical_features(
                     temp_X_processed,
                     col,
                     cardinality_threshold=cardinality_threshold
                 )
                 if temp_X_processed.empty: # Check if encoding removed everything
                      break

        X_processed = temp_X_processed # Assign result back

        print(f"\n**{fun_name}**: --- Categorical Encoding Finished ---")

        if X_processed.empty:
             print(f"**{fun_name}**: Warning: Feature DataFrame is empty after encoding.")
             return None # Return None if no features remain

        # Concatenate only if features exist
        final_df = pd.concat([X_processed, y], axis=1)

        print(f"\n**{fun_name}**: --- Full Preprocessing Finished ---")
        return final_df
        
    def _encode_categorical_features(
        self,
        dataframe: pd.DataFrame,
        column_name: str,
        cardinality_threshold: int = 10
        ) -> pd.DataFrame:
        """
        Encodes a single categorical column in the DataFrame.

        Chooses between One-Hot Encoding (for low cardinality) and
        Frequency Encoding (for high cardinality).
        """
        fun_name = "_encode_categorical_features"
        print(f"\n**{fun_name}**: Encoding column: '{column_name}'")
        
        if column_name not in dataframe.columns:
            print(f"**{fun_name}**: Warning: Column '{column_name}' not found. Skipping.")
            return dataframe
        
        series = dataframe[column_name]
        
        # Ensure the column is treated as categorical/object
        if not pd.api.types.is_object_dtype(series):
             print(f"**{fun_name}**: Warning: Column '{column_name}' is not object/category dtype \
                 ({series.dtype}). Skipping encoding.")
             return dataframe
         
         # Calculate cardinality (number of unique values)
        n_unique = series.nunique()
        print(f"**{fun_name}**:   Unique values: {n_unique}")
        
        try:
            if n_unique <= 1:
                print(f"**{fun_name}**: Action: Skipping encoding for '{column_name}' (single or no unique value).")
                dataframe = dataframe.drop(column_name, axis=1)
                print(f"**{fun_name}**: Dropped column '{column_name}' due to single unique value.")
                
            elif n_unique <= cardinality_threshold:
                print(f"**{fun_name}**: Action: Applying One-Hot Encoding to '{column_name}' (low cardinality).")
                dummies = pd.get_dummies(series, prefix=column_name, dummy_na=False) 
                dataframe = pd.concat([dataframe.drop(column_name, axis=1), dummies], axis=1)
                
            else:
                print(f"**{fun_name}**: Action: Applying Frequency Encoding to '{column_name}' (high cardinality).")
                frequency_map = series.map(series.value_counts(normalize=True))
                dataframe[column_name + '_freq'] = frequency_map
                dataframe = dataframe.drop(column_name, axis=1) 
        
        except Exception as e:
            print(f"**{fun_name}**: Error encoding column '{column_name}': {e}. Skipping.")
            
        return dataframe

    
    def run_baseline_pipeline(self, dataframe: pd.DataFrame, target: str) -> object | None:
        """
        Runs a baseline modeling pipeline: preprocess, split, train default model, evaluate.
        """
        fun_name = "run_baseline_pipeline"
        print(f"\n**{fun_name}**: --- Starting Baseline Pipeline ---")
        self.model = None # Reset model state at the start of a run

        # 1. Prepare data (Preprocess, Splitting)
        prepared_data = self._prepare_data_for_modeling(dataframe, target)
        if prepared_data is None:
            print(f"**{fun_name}**: Halting pipeline due to data preparation issues.")
            # self.model is already None from the start
            return None

        X_train, X_test, y_train, y_test, task_type = prepared_data

        # 2. Select and Train Model
        model = self._select_and_train_default_model(X_train, y_train, task_type)
        if model is None:
            print(f"**{fun_name}**: Halting pipeline due to model training failure.")
             # self.model is already None from the start
            return None

        # Assign model ONLY if training succeeded
        self.model = model

        # 3. Evaluate Model
        self._evaluate_model(self.model, X_test, y_test, task_type)

        print(f"**{fun_name}**: --- Baseline Pipeline Finished ---")
        return self.model
    
    def _prepare_data_for_modeling(
            self, dataframe: pd.DataFrame, target: str
            ) -> tuple | None:
        """
        Prepares data for modeling: determines task, preprocesses, splits.
        Returns (X_train, X_test, y_train, y_test, task_type) or None if errors occur.
        """
        fun_name = "_prepare_data_for_modeling"
        print(f"\n**{fun_name}**: --- Preparing Data ---")

        # 1a. Determine Task Type
        task_type = self.get_task_type(dataframe, target)
        if task_type is None:
             print(f"**{fun_name}**: Error: Could not determine task type for target '{target}'.")
             return None
        print(f"**{fun_name}**: Determined task type: {task_type}")


        # 1b. Preprocess Data
        processed_df = self.preprocess_data(dataframe, target, task_type)

        if processed_df is None:
            print(f"**{fun_name}**: Preprocessing resulted in no features or failed. Halting data preparation.")
            return None # Stop preparation immediately if no features

        # --- If we reach here, processed_df is a valid DataFrame with features + target ---

        # Check target integrity after potential NaN dropping during preprocessing (if it happened)
        if target not in processed_df.columns:
            print(f"**{fun_name}**: Internal Error: Target missing after successful preprocessing. Cannot proceed.")
            return None # Should be unlikely now

        # Drop rows with NaN in target (now safe as processed_df is valid)
        processed_df = processed_df.dropna(subset=[target])
        if processed_df.empty:
            print(f"**{fun_name}**: Warning: DataFrame is empty after dropping NaNs in target. Cannot proceed.")
            return None

        # 1c. Separate Features (X) and Target (y)
        # Since processed_df is not None and has target, X will have >= 0 features
        y = processed_df[target]
        X = processed_df.drop(target, axis=1)

        # Double check X shape (should be redundant now but safe)
        if X.shape[1] == 0:
             print(f"**{fun_name}**: Internal Warning: No features found after separation, though preprocessing returned a DataFrame.")
             return None

        # 1e. Split Data
        print(f"**{fun_name}**: Splitting data into training and testing sets...")
        try:
            split_data = self._split_data(X, y, task_type)
            if split_data is None:
                return None # Error already printed in _split_data

            X_train, X_test, y_train, y_test = split_data
            print(f"**{fun_name}**: Data preparation and split complete.")
            return X_train, X_test, y_train, y_test, task_type

        except Exception as e:
             print(f"**{fun_name}**: Unexpected error during data preparation's split phase: {e}")
             return None
        
    def _split_data(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> tuple | None:
        """Splits data into train/test sets with stratification for classification."""
        fun_name = "_split_data"
        try:
            # Use stratification for classification if possible
            # Note: stratificaiton is a special sampling technique
            stratify_param = y if task_type == 'classification' and y.nunique() > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                 X, y, test_size=0.2, random_state=42, stratify=stratify_param
             )
            
            print(f"**{fun_name}**: Data split complete. Train shape: {X_train.shape}, Test shape:\
                {X_test.shape}")
            return X_train, X_test, y_train, y_test
        
        except ValueError as e:
             print(f"**{fun_name}**: Warning during train/test split (possibly due to insufficient samples\
                 for stratification): {e}. Trying without stratification.")
             try:
                 X_train, X_test, y_train, y_test = train_test_split(
                     X, y, test_size=0.2, random_state=42
                 )
                 print(f"**{fun_name}**: Data split complete (without stratification). Train shape:\
                     {X_train.shape}, Test shape: {X_test.shape}")
                 return X_train, X_test, y_train, y_test
             except Exception as split_err:
                 print(f"**{fun_name}**: Error during train/test split even without stratification:\
                     {split_err}. Cannot proceed.")
                 return None
        except Exception as general_err:
             print(f"**{fun_name}**: Unexpected error during data split: {general_err}")
             return None
         
    def _select_and_train_default_model(self, X_train: pd.DataFrame, y_train: pd.Series, task_type: str) -> object | None:
        """Selects and trains a default model based on the task type."""
        fun_name = "_select_and_train_default_model"
        print(f"\n**{fun_name}**: --- Selecting and Training Model ---")

        # Select Model based on Task Type
        print(f"**{fun_name}**: Choosing the best default model for {task_type}...")
        if task_type == 'classification':
            chosen_model_name = 'LogisticRegression'
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif task_type == 'regression':
            chosen_model_name = 'LinearRegression'
            model = LinearRegression()
        else:
            print(f"**{fun_name}**: Error: Unsupported task type '{task_type}'.")
            return None

        print(f"**{fun_name}**: Model selected: {chosen_model_name}")

        # Train Model
        print (f"**{fun_name}**: Training {chosen_model_name} model...")
        try:
            model.fit(X_train, y_train)
            print(f"**{fun_name}**: Model trained successfully.")
            return model
        except Exception as e:
            print(f"**{fun_name}**: Error during model training: {e}")
            return None
        
    def _evaluate_model(self, model: object, X_test: pd.DataFrame, y_test: pd.Series, task_type: str) -> None:
        """Evaluates the trained model on the test set."""
        fun_name = "_evaluate_model"
        print(f"\n**{fun_name}**: --- Evaluating Model ---")
        print(f"**{fun_name}**: Evaluating model on the test set...")
        
        try:
            y_pred = model.predict(X_test)

            if task_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, zero_division=0)
                print(f"**{fun_name}**: Evaluation Results (Classification):")
                print(f"  Model Accuracy: {accuracy:.4f}")
                print(f"  Classification Report:\n{report}")

            elif task_type == 'regression':
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print(f"**{fun_name}**: Evaluation Results (Regression):")
                print(f"  Mean Squared Error: {mse:.4f}")
                print(f"  R-squared: {r2:.4f}")
                
        except Exception as e:
             print(f"**{fun_name}**: Error during model evaluation: {e}")
            
        
        
                
                
                
    
    