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
        
    def clean_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.dropna()
    
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
                 
                
                
                
    
    