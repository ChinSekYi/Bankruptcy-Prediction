import os
import sys
from pathlib import Path

import pandas as pd

# Add the project root to the PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.exception import CustomException
from src.logger import logging

print("works")

train_df = pd.read_csv("notebook/data/3year.csv")
