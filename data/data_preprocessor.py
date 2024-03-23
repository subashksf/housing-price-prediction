from logger import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor():
    def __init__(self):
        pass
    
    def drop_columns(self, dataframe, columns_to_drop):
        """
        Drop specified columns from the DataFrame.

        Parameters:
        - dataframe: DataFrame
            The input DataFrame.
        - columns_to_drop: list
            List of column names to drop.

        Returns:
        - DataFrame
            The DataFrame with specified columns dropped.
        """
        try:
            dataframe = dataframe.drop(columns_to_drop, axis=1)
            for col in columns_to_drop:
                logging.info(f"Dropped the column '{col}' from the dataset.")
        except KeyError as e:
            logging.error(f"Column '{e.args[0]}' not found in the DataFrame.")
        except Exception as e:
            logging.error(f"An exception occurred: {e}")
        return dataframe
    
    def fillna_mean(self, dataframe, columns_to_replace):
        """
        Fill missing values in DataFrame columns with the mean of each respective column.

        Parameters:
        - dataframe: DataFrame
            The input DataFrame.
        - columns_to_replace: list
            List of column names to consider.

        Returns:
        - DataFrame
            The DataFrame with missing values replaced by column means.
        """
        try:
            for col in columns_to_replace:
                col_mean = dataframe[col].mean()
                dataframe[col] = dataframe[col].fillna(col_mean)
                logging.info(f"Replaced missing values in the numeric column '{col}' with the mean.")
        except KeyError as e:
            logging.error(f"Column '{e.args[0]}' not found in the DataFrame.")
        except Exception as e:
            logging.error(f"An exception occurred: {e}")
        return dataframe

    def fillna_mode(self, dataframe, columns_to_replace):
        """
        Fill missing values in DataFrame columns with the mode of each respective column.

        Parameters:
        - dataframe: DataFrame
            The input DataFrame.
        - columns_to_replace: list
            List of column names to consider.

        Returns:
        - DataFrame
            The DataFrame with missing values replaced by column modes.
        """
        try:
            for col in columns_to_replace:
                col_mode = dataframe[col].mode().iloc[0]  # Mode might return multiple values, so take the first one
                if not pd.isna(col_mode):  # Check if mode exists before filling missing values
                    dataframe[col] = dataframe[col].fillna(col_mode)
                    logging.info(f"Replaced missing values in the categorical column '{col}' with the mode '{col_mode}'.")
                else:
                    logging.info(f"No mode found for categorical column '{col}'.")
        except KeyError as e:
            logging.error(f"Column '{e.args[0]}' not found in the DataFrame.")
        except Exception as e:
            logging.error(f"An exception occurred: {e}")
        return dataframe
    
    def label_encode(self, dataframe, columns_to_encode):
        """
        Label encode categorical columns in the DataFrame.

        Parameters:
        - dataframe: DataFrame
            The input DataFrame.
        - columns_to_encode: list
            List of column names to encode.

        Returns:
        - DataFrame
            The DataFrame with categorical columns label encoded.
        """
        try:
            label_encoder = LabelEncoder()
            for col in columns_to_encode:
                dataframe[col] = label_encoder.fit_transform(dataframe[col])
                logging.info(f"Label encoded categorical column '{col}'.")
        except Exception as e:
            logging.error(f"An exception occurred: {e}")
        return dataframe
