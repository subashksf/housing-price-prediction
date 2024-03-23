from sklearn.model_selection import train_test_split
from logger import logging

class Trainer():
    def __init__(self):
        pass
    
    def perform_train_test_split(self, data, target_column):
        """
        Split the dataset into training and testing sets.

        Parameters:
        - data: DataFrame
            The input DataFrame containing the dataset.
        - target_column: str
            The name of the target column.

        Returns:
        - tuple
            A tuple containing X_train, X_test, y_train, and y_test.
        """
        try:
            X = data.drop(target_column, axis=1)
            y = data[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            logging.info("Performed train-test split successfully.")
            logging.info(f"Shape of X_train data: {X_train.shape}")
            logging.info(f"Shape of y_train data: {y_train.shape}")
            logging.info(f"Shape of X_test data: {X_test.shape}")
            logging.info(f"Shape of y_test data: {y_test.shape}")
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"An error occurred during train-test split: {e}")
            return None, None, None, None
