from typing import Dict
import json
import pandas as pd
import numpy as np
import clearbox_wrapper as cbw

from fastapi import APIRouter, HTTPException

router = APIRouter()
model = cbw.load_model("/app/resources")


@router.post("/predict")
async def predict(instance: Dict):
    instance_df = pd.Series(instance).to_frame().T
    try:
        try:
            prediction = model.predict_proba(
                instance_df)
        except cbw.ClearboxWrapperException:
            prediction = model.predict(
                instance_df)
    except Exception:
        raise HTTPException(
            status_code=422, detail="Unprocessable Entity. Please provide a valid input.")

    return {
        "RESULT": json.dumps(prediction, cls=NumpyEncoder)
    }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
