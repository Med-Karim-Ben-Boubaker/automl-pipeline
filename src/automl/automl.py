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
        
    def get_task_type(self, dataframe: pd.DataFrame, target: str) -> str:
        target_series = dataframe[target]
        
        if pd.api.types.is_numeric_dtype(target_series):
            if target_series.nunique() <= 10:
                self.task_type = 'classification'
                return 'classification'
            else:
                self.task_type = 'regression'
                return 'regression'
        
        else:
            self.task_type = 'classification'
            return 'classification'
        
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
        fun_name = "preprocess_data"
        
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
    ) -> pd.DataFrame:
        
        return
        
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

    
    def train_model(self, dataframe: pd.DataFrame, target: str, type: str) -> None:
        
        dataframe = dataframe.dropna()

        print("Choosing the best model for classification...")
        
        if type == 'classification':

            chosen_model = 'LogisticRegression'
        elif type == 'regression':

            chosen_model = 'LinearRegression'
        
        print(f"Model chosen: {chosen_model}")
        
        print("Initializing model...")
        
        # Seperate features (X) and target (y)
        y = dataframe[target]
        X = dataframe.drop(target, axis=1)
        
        # Select only numerical columns
        numerical_cols = X.select_dtypes(include=['number']).columns
        X_numerical = X[numerical_cols]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_numerical, y, test_size=0.2, random_state=42
            )
        
        print ("Training model...")
        if chosen_model == 'LogisticRegression':
            
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            self.model = model
            
        elif chosen_model == 'LinearRegression':

            model = LinearRegression()
            model.fit(X_train, y_train)
            
            self.model = model
            
        print("Model trained successfully.")
            
        # Make predictions on the test set
        y_pred = model.predict(X_test)

        
        if type == 'classification':
            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            print(f"Model accuracy: {accuracy}")
            print(f"Classification report:\n{report}")
            
        elif type == 'regression':
            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Mean Squared Error: {mse}")
            print(f"R-squared: {r2}")
        
        
        return model
                 
                
                
                
    
    