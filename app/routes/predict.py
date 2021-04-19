from typing import Dict, List
import json
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
from fastapi import APIRouter


router = APIRouter()
model = joblib.load("../model.pkl")

class InputExample(BaseModel):
    age: int = 32
    work_class: str = 'Local-gov'
    education: str = 'Masters'
    marital_status: str = 'Married'
    occupation: str = 'Professional'
    relationship: str = 'Wife'
    race: str = 'White'
    sex: str = 'Female'
    capital_gain: int = 0
    capital_loss: int = 0
    hours_per_week: int = 28
    native_country: str = 'United-States'

@router.post("/predict")
async def predict(instance: InputExample):
    instance_df = pd.DataFrame(instance).T
    instance_df.columns = instance_df.iloc[0]
    instance_df = instance_df[1:]
    prediction = model.predict_proba(instance_df)

    return {
        "RESULT": json.dumps(prediction, cls=NumpyEncoder)
    }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)