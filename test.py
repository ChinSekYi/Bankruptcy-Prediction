import logging
import os
import sys
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

print(LOG_FILE)
print(logs_path)
print(LOG_FILE_PATH)
