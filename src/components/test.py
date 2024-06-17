import os
import sys
from pathlib import Path

# Add the project root to the PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.exception import CustomException
from src.logger import logging

print("works")

print([i for i in range(0, 64)])
