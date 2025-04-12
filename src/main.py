from automl.automl import AutoMLRunner

def main():
    automl = AutoMLRunner()

    df = automl.load_csv_data('/home/karim/automl-pipeline/data/regression_dataset.csv')
    
    print(f"DataFrame loaded: {df.head()}")
    
    type = automl.get_task_type(df, 'target')
    
    print(f"Task type: {type}")
    
    trained_model = automl.train_model(df, 'target', type)
    
if __name__ == "__main__":
    main()