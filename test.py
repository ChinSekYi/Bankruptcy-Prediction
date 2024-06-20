import logging
import os
import sys
from datetime import datetime
from src.utils import save_object, load_object

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

#load_object('/Users/sekyichin/Downloads/Bankruptcy-Prediction/artifacts/model.pkl')
load_object('artifacts/preprocessor.pkl')
#load_object('artifacts\\model.pkl') s