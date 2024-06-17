import sys
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.exception import CustomException
from src.logger import logging


class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artefacts', 'preprocessor.pk1')