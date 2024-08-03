import os
import sys
import dill
import pickle
import numpy as np
import pandas as pd

from src.exception import CustomException

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise CustomException(str(e))