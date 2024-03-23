import pandas as pd 
from logger import logging

class DataLoader:
    def __init__(self):
        pass

    def load_data(self):
        try:
            data = pd.read_csv('data/train.csv', encoding='utf-8')
        except Exception as e:
            logging.Error("Error: ", e)

        return data