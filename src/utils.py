"""
utils.py

This module contains utility functions for handling object serialization,
model evaluation, and printing bankruptcy prediction outcomes.
"""

import os
import pickle
import sys

import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from exception import CustomException


def save_object(file_path, obj):
    """
    Save an object to a file using pickle.

    Args:
        file_path (str): Path to the file where the object will be saved.
        obj (object): Object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e


def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    """
    Evaluate multiple models using GridSearchCV and return their R-squared scores.

    Args:
        x_train (array-like): Training input samples.
        y_train (array-like): Target values for training.
        x_test (array-like): Test input samples.
        y_test (array-like): Target values for testing.
        models (dict): Dictionary of models to evaluate.
        params (dict): Dictionary of parameter grids for GridSearchCV.

    Returns:
        dict: Dictionary containing model names as keys and their R-squared scores as values.
    """
    try:

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            # y_train_pred = model.predict(x_train)

            y_test_pred = model.predict(x_test)

            # train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path):
    """
    Load an object from a file using dill.

    Args:
        file_path (str): Path to the file containing the object.

    Returns:
        object: Loaded object.
    """
    try:
        print(file_path)
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e


def print_bankruptcy_outcome(pred_result):
    """
    Print the bankruptcy prediction outcome based on the prediction result.

    Args:
        pred_result (float): Predicted probability or score.

    Returns:
        str: Prediction outcome message.
    """
    if pred_result >= 0.5:
        return "Bad news! The company is predicted to be bankrupt within 3 years."
    return (
        "Good news! The company is "
        "predicted to continue operating successfully for the next 3 years."
    )
