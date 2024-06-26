import logging
import os
import sys
from datetime import datetime

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import load_object, print_bankruptcy_outcome, save_object

# TODO debug
preprocessor = load_object(file_path=os.path.join("artifacts", "preprocessor.pkl"))
print(preprocessor)

data = CustomData(
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
)

pred_df = data.get_data_as_dataframe()
print(f"\033[1;31;40m pred_df  \033[0m 1;31;40m   ")
print(pred_df)

predict_pipeline = PredictPipeline()
# results = predict_pipeline.predict(pred_df)

model_path = os.path.join("artifacts", "model.pkl")
preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
model = load_object(file_path=model_path)
preprocessor = load_object(file_path=preprocessor_path)
data_scaled = preprocessor.transform(pred_df)
pred_result = model.predict(data_scaled)

print(f"Prediction result: {pred_result}")
print(print_bankruptcy_outcome(pred_result))
