from logger import logging
import pandas as pd

class Data_Preprocessor():
    def __init__(self):
        pass
    
    # This function drops the columns as determined after EDA
    def drop_columns(self, dataframe, columns_to_drop):
        for col in columns_to_drop:
            dataframe = dataframe.drop(col, axis = 1)
            logging.info(f"Dropped the column {col} from the dataset.")

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
        except Exception as e:
            logging.error(f"Exception occurred: {e}")
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
