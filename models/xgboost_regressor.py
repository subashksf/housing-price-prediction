from logger import logging
from train.trainer import Trainer
from evaluate.evaluator import Evaluator
from xgboost import XGBRegressor

trainer = Trainer()
evaluator = Evaluator()

class XGBoostRegressor():
    def __init__(self):
        pass
    
    def model_xgboost_regressor(self, data, target_column):
        """
        Train an XGBoost regressor model on the given data and return predictions.

        Parameters:
        - data: DataFrame
            The input DataFrame containing the dataset.
        - target_column: str
            The name of the target column.

        Returns:
        - numpy array
            Predictions made by the trained model.
        """
        try:
            # Train test split
            X_train, X_test, y_train, y_test = trainer.perform_train_test_split(data, target_column)

            # Define the model using the best parameters are determined by GridSearchCV (from the notebook)
            model_xgb = XGBRegressor(learning_rate = 0.015,
                            n_estimators  = 1000,
                            max_depth     = 5,
                            eval_metric='rmsle')
          
            # Train the refined model
            model_xgb.fit(X_train, y_train)

            # Make predictions on the test data
            predictions = model_xgb.predict(X_test)
            logging.info("XGBoost regressor model trained successfully.")

            # Evaluating the model
            evaluator.evaluate_model("XGBoostRegressor", predictions=predictions, y_test=y_test)
            logging.info("XGBoost regressor model evaluated successfully.")

            return predictions
        
        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            return None
        