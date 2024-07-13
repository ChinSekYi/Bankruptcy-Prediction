"""
Module for training various machine learning models and evaluating their performance.
"""

import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, save_object


@dataclass
class ModelTrainerConfig:
    """
    Configuration class for model training operations.

    Attributes:
    - trained_model_file_path (str): File path to save the trained model.
    """

    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    ModelTrainer class handles training of various machine learning models and
    evaluation of their performance.

    Methods:
    - __init__(): Initializes a ModelTrainer instance with default configuration.
    - initiate_model_trainer(train_array, test_array): Initiates model training,
    evaluates model performance, and saves the best model.

    """

    def __init__(self):
        """
        Initializes a ModelTrainer instance with default configuration.

        Usage:
        >>> trainer = ModelTrainer()
        """
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Initiates model training, evaluation, and saves the best performing model.

        Args:
        - train_array (numpy.ndarray): Array containing training data.
        - test_array (numpy.ndarray): Array containing testing data.

        Returns:
        - Tuple: Predicted values and R-squared score of the best performing model.

        Raises:
        - CustomException: If an error occurs during model training or evaluation.

        Usage:
        >>> trainer = ModelTrainer()
        >>> predicted_values, r2_score = trainer.initiate_model_trainer(train_array, test_array)
        """
        try:
            logging.info("Splitting training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {"Linear Regression": LinearRegression()}

            """
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNN": KNeighborsRegressor(),
                "SVM": SVM(),
                "XGBClassifier": XGBRegressor(),
                "AdaBoosting Classifier": AdaBoostRegressor(),
            } """

            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                },
                "Random Forest": {"n_estimators": [8, 16, 32, 64, 128, 256]},
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.01],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "Logistic Regression": {},
                "KNN": {},
                "SVM": {},
                "XGBClassifier": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "AdaBoosting Classifier": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            model_report: dict = evaluate_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # TODO: Add a threshold for best model score
            """
            if best_model_score < 0.6:
                error_messsage = "No best model found"
                logging.error(error_messsage)
                raise Exception(error_messsage)
            """

            logging.info(
                f"Best model is {best_model_name} with score: {best_model_score} on testing dataset"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(x_test)

            r2_score_value = r2_score(y_test, predicted)

            return (predicted, r2_score_value)

        except Exception as e:
            raise CustomException(e, sys) from e
