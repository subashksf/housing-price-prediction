from logger import logging
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_log_error

class Evaluator():
    def __init__(self):
        pass

    def evaluate_model(self, model, predictions, y_test):
        RMSLE_xgb = np.sqrt( mean_squared_log_error(y_test, predictions) )
        logging.info(f"The RMSE score for the model {model} is %.5f (XGB)" % RMSLE_xgb )

        mse_xgb = mean_squared_error(y_test, predictions)
        logging.info(f'Mean Squared Error for the model {model} (XGB): {mse_xgb}')

        r2_xgb = r2_score(y_test, predictions)
        logging.info(f'R-squared for the model {model}: {r2_xgb}')