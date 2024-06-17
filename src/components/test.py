import os
import sys
from pathlib import Path

# Add the project root to the PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

print(os.path.abspath("src"))
print(os.listdir(project_root))

from src.exception import CustomException
from src.logger import logging

print("works")
