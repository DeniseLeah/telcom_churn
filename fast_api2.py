import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from graphviz import pipe_lines
from mlflow import catboost
from mlflow.metrics import f1_score
from pandas.core.common import random_state
from scipy.constants import precision
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy.sql.util import criterion_as_pairs
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Binarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
pd.options.display.max_columns = None
sklearn.set_config(transform_output='pandas')
from sklearn.metrics import average_precision_score
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
mlflow.set_registry_uri('./mlruns')
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import phik
import joblib
from pydantic import BaseModel
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from fastapi.responses import JSONResponse

# Load models
model_lr = joblib.load("lr_model.pkl")
model_cat = joblib.load("cat_model.pkl")

# FastAPI app
app = FastAPI()

# Define input data model
class Data_to_predict(BaseModel):
    data: List[Dict]

@app.post("/predict")
async def predict(dpp: Data_to_predict):
    try:
        df = pd.DataFrame(dpp.data)
        preds = model_lr.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict_cat")
async def predict_cat(dpp: Data_to_predict):
    try:
        df = pd.DataFrame(dpp.data)
        preds = model_cat.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
