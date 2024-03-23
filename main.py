from logger import logging
from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from train.trainer import Trainer
import pandas as pd

def main():
    logging.info("Starting the program..")
    data_loader = DataLoader()
    data_preprocessor = DataPreprocessor()
    trainer = Trainer()
    
    # Load the dataset  
    df = data_loader.load_data()

    # Adjust the display settings
    pd.set_option('display.max_rows', None)  # Set to None to display all rows

    # Drop the columns determined after EDA
    logging.info("Dropping the columns as determined by EDA")
    columns_to_drop = ['Id','GarageYrBlt','Alley','PoolQC','Fence','MiscFeature']
    df = data_preprocessor.drop_columns(df, columns_to_drop)

    # Replace null values of numeric columns with the mean of the column
    logging.info("Handling null values in numeric columns")

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['number'])
    
    # Get numeric columns with null values
    numeric_cols_with_nulls = numeric_cols.columns[numeric_cols.isnull().any()]
    logging.info("Numeric columns with null values: %s", numeric_cols_with_nulls)

    # Replace null values with mean
    df = data_preprocessor.fillna_mean(df, numeric_cols_with_nulls)

    # Select categorical columns
    categorical_cols = df.select_dtypes(include = ['object'])
    
    # Get categorical columns with null values
    categorical_cols_with_nulls = categorical_cols.columns[categorical_cols.isnull().any()]
    logging.info("Categorical columns with null values: %s", categorical_cols_with_nulls)
    
    # Replace null values with mode
    df = data_preprocessor.fillna_mode(df, categorical_cols_with_nulls)

    # Verify the dataset for any null values after the proprocessing
    logging.info(df.isna().sum())

    # Label encode categorical columns
    df = data_preprocessor.label_encode(df, categorical_cols)

    # Perform train test split
    X_train, X_test, y_train, y_test = trainer.perform_train_test_split(df, 'SalePrice')

if __name__ == "__main__":
    main()