import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])
    
install('gensim==4.3.2')
install('rapidfuzz==3.6.1')
install('scikit-learn==1.0.2')
install('category_encoders==2.6.3')


import argparse
import pathlib
import boto3
import logging
import os
import json
import joblib
import pickle
import re
import ast
import tarfile
import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
from io import StringIO
from ast import literal_eval
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sagemaker_containers.beta.framework import worker
from sklearn.metrics import roc_auc_score
from category_encoders import OrdinalEncoder
from utils.transform_features import *

additonal_cols = ['job_allows_easy_apply',
                  'job_open_for_career_changers',
                  'jobseeker_preference_salary_min',
                  'job_salary_amount',
                  'job_salary_range_min',
                  'job_salary_range_max'
                 ]

# Create a list of columns to drop from the dataset
columns_to_drop = ['jdp_view_count', 
                   'job_bookmark_count', 
                   'application_start_count', 
                   'application_sent_count', 
                   'application_submit_count', 
                   'gold_application_sent_count']

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-jobseeker-uid-in-training", type=str, default="yes")
    
    args = parser.parse_args()
    logger.info("Loaded parser arguments")
    
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading xgboost model.")
    # The name of the file should match how the model was saved in the training script
    # model = pickle.load(open("xgboost-model", "rb"))
    model = xgb.Booster()
    model.load_model('xgboost-model')
    
    logger.debug("Loading preprocess transformer.")
    # The name of the file should match how the model was saved in the training script
    preprocess_transformer_path = "/opt/ml/processing/preprocess_pipeline"
    preprocessor = joblib.load(os.path.join(preprocess_transformer_path, "preprocess_pipeline.joblib"))

    logger.debug("Reading test data.")
    test_local_path = "/opt/ml/processing/test/test.parquet"
    data_test = pd.read_parquet(test_local_path)
    
    data_test = process_dataframe(data_test)
    data_test = data_test.drop(columns=columns_to_drop, axis=1)
    
    # Extract test set target column
    y_test = data_test['target'].values
    X_test = pd.DataFrame(preprocessor.transform(data_test))
    # Concatenate X_train and y_train
    X_test = pd.concat([X_test, data_test[additonal_cols]],
                       axis=1, ignore_index=True)
    X_test = X_test.to_numpy()
    
    X_test = xgb.DMatrix(X_test)

    logger.info("Generating predictions for test data.")
    pred = model.predict(X_test)
    
    # Calculate model evaluation score
    logger.debug("Calculating ROC-AUC score.")
    auc = roc_auc_score(y_test, pred)
    metric_dict = {
        "classification_metrics": {"roc_auc": {"value": auc}}
    }
    
    # Save model evaluation metrics
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing evaluation report with ROC-AUC: %f", auc)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(metric_dict))
