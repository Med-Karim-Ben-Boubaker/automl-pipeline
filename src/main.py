from automl.automl import AutoMLRunner

def main():
    automl = AutoMLRunner()

    file_path = '/home/karim/automl-pipeline/data/titanic/train.csv'
    df = automl.load_csv_data(file_path)

    if df is None:
        print("Failed to load data. Exiting.")
        return

    print(f"DataFrame loaded successfully from: {file_path}")

    target_column = 'Survived'
    task_type = automl.get_task_type(df, target_column)
    print(f"Detected task type: {task_type}")

    print("Starting data preprocessing...")
    
    preprocessed_df = automl.preprocess_data(
        df.copy(),
        target=target_column,
        task_type=task_type
    )
    print("Data preprocessing finished.")

    if preprocessed_df is not None and not preprocessed_df.empty:

        output_file_path = '/home/karim/automl-pipeline/data/cleaned_titanic.csv'
        preprocessed_df.to_csv(output_file_path, index=False)
        print(f"Preprocessed data saved to: {output_file_path}")
    else:
        print("Preprocessing resulted in an empty DataFrame or an error. No CSV file saved.")
        if preprocessed_df is None:
            print("Preprocessing returned None.")
        elif preprocessed_df.empty:
            print("Preprocessing returned an empty DataFrame.")

if __name__ == "__main__":
    main()