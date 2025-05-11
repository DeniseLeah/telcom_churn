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




model_lr = joblib.load('lr_model.pkl')
model_cat = joblib.load('cat_model.pkl')

from fastapi import FastAPI
app = FastAPI()

class Data_to_predict(BaseModel): #description of the input data
    """
    Data model for input data.
    """
    {
        "data": "[{\"feature1\": 1, \"feature2\": 2}]"
    }
#input data in JSON format
@app.post("/predict") #endpoint for prediction()
async def predict(dpp: Data_to_predict): #endpoint for prediction
    """
    Predict the target variable using the loaded model.
    """
    # Convert the input data to a DataFrame
    df = pd.read_json(dpp.data, orient='records')

    # Make predictions
    predictions = model_lr.predict(df)

    # Return the predictions as a JSON response
    return {"predictions": predictions.tolist()}
# @app.get("/")
@app.post("/predict_cat") #endpoint for prediction()
async def predict_cat(dpp: Data_to_predict): #endpoint for prediction
    """
    Predict the target variable using the loaded model.
    """
    # Convert the input data to a DataFrame
    df = pd.read_json(dpp.data, orient='records')

    # Make predictions
    predictions = model_cat.predict(df)

    # Return the predictions as a JSON response
    return {"predictions": predictions.tolist()}