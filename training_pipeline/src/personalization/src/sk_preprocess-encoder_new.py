import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])
    
install('gensim==4.3.2')
install('rapidfuzz==3.6.1')
install('category_encoders==2.6.3')

import argparse
import pathlib
import boto3
import logging
import os
import json
import joblib
import re
import tarfile
import pandas as pd
import numpy as np
import datetime
import warnings
from io import StringIO
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sagemaker_containers.beta.framework import worker
from category_encoders import OrdinalEncoder
from utils.transform_features import *

try:
    from sagemaker_containers.beta.framework import (
        encoders,
        worker,
    )
except ImportError:
    pass

# suppress only the 'Mean of empty slice' RuntimeWarning:
warnings.filterwarnings(action='ignore', message='Mean of empty slice', category=RuntimeWarning)

logger = logging.getLogger()
logger.setLevel(logging.WARN)
logger.addHandler(logging.StreamHandler())


cat_cols_imp_ns = [
    # 'last_browser_family',
    # 'last_device_type',
    # 'job_product_type',
    'jobseeker_preference_job_type_codename',
    # 'jobseeker_preference_salary_period',
    # 'job_salary_period',
    # 'job_origin',
    # 'job_link_out_type',
    'job_required_experience',
    'job_required_education',
    # 'job_shift',
    'job_company_uid'
]

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

label_features = ['target', 
                  'target_w2v']

# these features are for random negative sample creation
job_features = ['job_uid', 'job_product_type', 'job_company_uid', 'job_title',
               'job_description', 'job_salary_amount', 'job_salary_currency',
               'job_salary_period', 'job_origin', 'job_working_hours',
               'job_employment_types', 'job_city', 'job_lat', 'job_lng',
               'job_allows_easy_apply', 'job_salary_range_min', 'job_salary_range_max',
               'job_salary_incentive', 'job_link_out_type', 'job_working_from_home',
               'job_required_experience', 'job_required_education',
               'job_open_for_career_changers', 'job_schedule_type', 'job_shift',
               'job_type_de_name', 'job_kldb_code']


def find_latest_training_data(s3_client, bucket_name, prefix, date_format="%Y-%m-%d"):
    file_list = []
    interactions = {'positive_interactions': {}, 'negative_interactions': {}}
    
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            file_list.append(obj.get("Key"))
    
    for file_path in file_list:
        parts = file_path.split("/")
        interaction_type = parts[3]
        date_str = parts[4].split("=")[1]
        date = datetime.datetime.strptime(date_str, date_format)

        if date not in interactions[interaction_type]:
            interactions[interaction_type][date] = file_path


    common_dates = set(interactions['positive_interactions']).intersection(set(interactions['negative_interactions']))
    
    if not common_dates:
        return []
    
    latest_date = max(common_dates)
    return [
        interactions['positive_interactions'][latest_date],
        interactions['negative_interactions'][latest_date]
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-validation-ratio", type=float, default=0.9)
    parser.add_argument("--test-day-span", type=int, default=5)
    parser.add_argument("--negative-sample-ratio", type=int, default=2)
    parser.add_argument("--random-negative-sample-multiplier", type=int, default=2)
    parser.add_argument("--include-jobseeker-uid-in-training", type=str, default="no")
    parser.add_argument("--include-jdp-view-as-target", type=str, default="no")
    args, _ = parser.parse_known_args()
    
    train_val_ratio = args.train_validation_ratio
    test_day_span = args.test_day_span
    negative_sample_ratio = args.negative_sample_ratio
    random_negative_sample_multiplier = args.random_negative_sample_multiplier
    include_jobseeker_uid_in_training = args.include_jobseeker_uid_in_training
    include_jdp_view_as_target = args.include_jdp_view_as_target
    
    logger.info("Received arguments {}".format(args))

    # Set local path prefix in the processing container
    local_dir = "/opt/ml/processing"
    raw_data_positive = "positive.csv"
    raw_data_negative = "negative.csv"
    
    s3_client = boto3.client("s3")
    bucket_name = "sagemaker-personalization-mlops-poc"
    latest_data = find_latest_training_data(
        s3_client,
        bucket_name=bucket_name,
        prefix="search_feed"
    )
    logger.info(latest_data)
    if len(latest_data) > 0:
        s3_client.download_file(bucket_name, latest_data[0], raw_data_positive)
        s3_client.download_file(bucket_name, latest_data[1], raw_data_negative)
        logger.info("Downloaded latest data to local")
    else:
        logger.error("New data not found")

    logger.info("Reading data from {} & {}".format(raw_data_positive, raw_data_negative))
    negative_df = pd.read_csv(raw_data_negative)
    positive_df = pd.read_csv(raw_data_positive)
    
    # convert to date format
    positive_df['derived_tstamp'] = pd.to_datetime(positive_df['derived_tstamp'], 
                                                   format="%Y-%m-%d %H:%M:%S.%f",
                                                   errors='coerce').dt.date
    negative_df['derived_tstamp'] = pd.to_datetime(negative_df['derived_tstamp'], 
                                                   format="%Y-%m-%d %H:%M:%S.%f", 
                                                   errors='coerce').dt.date

    # Drop rows with missing derived_tstamp
    positive_df = positive_df.dropna(subset=['derived_tstamp'])
    negative_df = negative_df.dropna(subset=['derived_tstamp'])
    
    logger.info(f"Generating test dataset by including last {test_day_span} days from main dataset")

    # split sample into train-validation & test
    n_days_ago = (positive_df['derived_tstamp'].max() - datetime.timedelta(days=test_day_span))

    positive_df_test = positive_df[positive_df['derived_tstamp'] > n_days_ago]
    negative_df_test = negative_df[negative_df['derived_tstamp'] > n_days_ago]

    # creating test data
    df_test = pd.concat([positive_df_test, negative_df_test], ignore_index=True)
    
    if include_jdp_view_as_target == "yes":
        logger.info("Including jdp_view_count as part of the test target labels")
        df_test.loc[:,'target'] = np.where(((df_test['jdp_view_count']>0) | 
                                            (df_test['job_bookmark_count']>0) |
                                               (df_test['application_start_count']>0) | 
                                                (df_test['application_sent_count']>0) |
                                                 (df_test['application_submit_count']>0) | 
                                                  (df_test['gold_application_sent_count']>0))
                                                    ,1,0)
    else:
        logger.info("Excluding jdp_view_count as part of the test target labels")
        df_test.loc[:,'target'] = np.where(((df_test['job_bookmark_count']>0) |
                                           (df_test['application_start_count']>0) | 
                                            (df_test['application_sent_count']>0) |
                                             (df_test['application_submit_count']>0) | 
                                              (df_test['gold_application_sent_count']>0))
                                                ,1,0)

    logger.info(f"Generating train-validation dataset by filtering out last {test_day_span} days from main dataset")
    positive_df = positive_df[positive_df['derived_tstamp'] <= n_days_ago]
    negative_df = negative_df[negative_df['derived_tstamp'] <= n_days_ago]

    # getting a pool of all the jobs
    combined_df = pd.concat([positive_df, negative_df], ignore_index=True)

    # create target variable in positive samples
    if include_jdp_view_as_target == "yes":
        logger.info("Including jdp_view_count as part of the train target labels")
        positive_df.loc[:,'target'] = np.where(((positive_df['jdp_view_count']>0) | 
                                                (positive_df['job_bookmark_count']>0) |
                                                   (positive_df['application_start_count']>0) | 
                                                    (positive_df['application_sent_count']>0) |
                                                     (positive_df['application_submit_count']>0) | 
                                                      (positive_df['gold_application_sent_count']>0))
                                                        ,1,0)
    else:
        logger.info("Excluding jdp_view_count as part of the train target labels")
        positive_df.loc[:,'target'] = np.where(((positive_df['job_bookmark_count']>0) |
                                               (positive_df['application_start_count']>0) | 
                                                (positive_df['application_sent_count']>0) |
                                                 (positive_df['application_submit_count']>0) | 
                                                  (positive_df['gold_application_sent_count']>0))
                                                    ,1,0)

    # creating extra target_w2v which will be used by word2vec model
    positive_df.loc[:,'target_w2v'] = np.where(((positive_df['jdp_view_count']>0) | 
                                              (positive_df['job_bookmark_count']>0) |
                                               (positive_df['application_start_count']>0) | 
                                                (positive_df['application_sent_count']>0) |
                                                 (positive_df['application_submit_count']>0) | 
                                                  (positive_df['gold_application_sent_count']>0))
                                                    ,1,0)

    # Drop the columns from the DataFrame
    positive_df = positive_df.drop(columns=columns_to_drop, axis=1)
    
    logger.info(f"Generating negative samples by taking at least {negative_sample_ratio} negative per positive sample per user per session from the respective positive session")
    # Filter negative sample just to take 
    negative_df = negative_df.groupby(['jobseeker_uid', 'derived_tstamp', 'session_key']
                                            ).head(negative_sample_ratio).reset_index(drop=True)
    negative_df = negative_df.drop(columns=columns_to_drop, axis=1)

    # directly assigning target and target_w2v as 0 being negative sample
    negative_df['target'] = 0
    negative_df['target_w2v'] = 0

    # filtering out only negative samples from a positively interacted session
    negative_df = pd.merge(positive_df.loc[positive_df['target']>0,
                                                                ['jobseeker_uid', 
                                                                 'derived_tstamp', 
                                                                 'session_key']].drop_duplicates(),
                                                                     negative_df, 
                                                                         on=['jobseeker_uid', 
                                                                             'derived_tstamp',
                                                                             'session_key'], 
                                                                         how='inner')

    # combining the positve and negative samples
    df_with_neg_samples = pd.concat([positive_df, negative_df], ignore_index=True)

    # Keep only the last occurrence for each unique user_id and job_uid combination (positive first)
    df_with_neg_samples.sort_values(by=['jobseeker_uid', 'job_uid', 'target', 'derived_tstamp'],
                                    ascending=[False, False, False, False], 
                                    inplace=True)
    df_with_neg_samples = df_with_neg_samples.drop_duplicates(subset=['jobseeker_uid', 'job_uid'], 
                                                              keep='first')
    
    logger.info(f"Generating {random_negative_sample_multiplier} time random negative sample for each unique user interaction (postive + negative)")
    # generating random negative samples of the size of the positive sample data
    user_features = df_with_neg_samples.drop(job_features + label_features, 
                                             axis=1).drop_duplicates().reset_index(drop=True)
    user_features = [user_features] * random_negative_sample_multiplier
    user_features = pd.concat(user_features, ignore_index=True)

    # Shuffle the rows of the large job features DataFrame based on N times
    # the number of unique user-sessions
    # removing the jobs which are part of the test dataset
    combined_df = combined_df.sample(n=len(user_features)).reset_index(drop=True)
    combined_df = combined_df[job_features]

    combined_df['target'] = 0
    combined_df['target_w2v'] = 0

    # pairing up random user sessions with random user jobs
    df_with_random_neg = pd.concat([user_features, combined_df], axis=1)

    # concatenating the dataframe with the random negative samples
    df_with_random_neg = pd.concat([df_with_neg_samples, 
                                    df_with_random_neg], 
                                   ignore_index=True)

    # Keep only the last occurrence for each unique user_id and job_uid combination (positive first)
    df_with_random_neg.sort_values(by=['jobseeker_uid', 'job_uid', 'target', 'derived_tstamp'],
                                    ascending=[False, False, False, False], 
                                    inplace=True)
    df_with_random_neg = df_with_random_neg.drop_duplicates(subset=['jobseeker_uid', 'job_uid'], 
                                                              keep='first')
    
    logger.info(f"Shape of positive & negative dataset i.e. df_with_neg_samples is: {df_with_neg_samples.shape}")
    logger.info(f"Distribution of positive & negative dataset i.e. df_with_neg_samples is:: {df_with_neg_samples.target.value_counts()}")
    logger.info(f"Shape of positive & negative dataset with random negative samples i.e. df_with_random_neg is: {df_with_random_neg.shape}")
    logger.info(f"Distribution of positive & negative dataset with random negative samples i.e. df_with_random_neg is: {df_with_random_neg.target.value_counts()}")
    
    logger.info(f"Building sklearn column transformer")
    # defining sklearn preprocessing pipeline
    cat_cols_pipeline_ns = Pipeline(steps=[
        ('encoders', OrdinalEncoder(handle_unknown='value', handle_missing='value'))
    ])

    # Pipeline for categorical columns with 'no' imputation
    cat_cols_pipeline_no = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='no')),
        ('encoders', OrdinalEncoder(handle_unknown='value', handle_missing='value'))
    ])

    # Pipeline for categorical columns with 'fixed_working_hours' imputation
    cat_cols_pipeline_fwh = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='fixed_working_hours')), 
        ('encoders', OrdinalEncoder(handle_unknown='value', handle_missing='value'))
    ])

    if include_jobseeker_uid_in_training == "yes":
        logger.info("Including jobseeker_uid as part of the training features")
        cat_cols_imp_ns = cat_cols_imp_ns + ['jobseeker_uid']
    
    # Define column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat_cols_ns', cat_cols_pipeline_ns, cat_cols_imp_ns),
            ('cat_cols_no', cat_cols_pipeline_no, ['job_working_from_home']),
            ('cat_cols_fwh', cat_cols_pipeline_fwh, ['job_schedule_type']),
            ('distance_calculator', DistanceCalculator(), ['jobseeker_preference_lng',
                                                          'jobseeker_preference_lat',
                                                          'job_lng',
                                                          'job_lat']),
            ('job_employment_types', MultiLabelBinarizerTransformer(), ['jobseeker_preference_employment_types',
                                                                        'common_employment_types', 
                                                                        'jobseeker_preference_working_hours', 
                                                                        'common_working_hours']),
            ('similarity_score_calculator', SimilarityScoreCalculator(), ['jobseeker_uid', 
                                                                          'jobseeker_preference_job_title_b2c', 
                                                                          'last_feed_search_query', 
                                                                          'job_title']),
        ], 
        verbose_feature_names_out=False
    )
    
    logger.info(f"Building train and validation dataset and spliting based on {train_val_ratio} ratio based on derived_tstamp")
    df_with_random_neg = df_with_random_neg[df_with_random_neg['target'] == df_with_random_neg['target_w2v']].reset_index(drop=True)
    
    # Identify users with past interactions
    df_with_random_neg =df_with_random_neg.sort_values(by='derived_tstamp', ascending=True)
    
    # perform general preprocessing of the dataframe
    df_with_random_neg = process_dataframe(df_with_random_neg)

    split_point = int(len(df_with_random_neg) * train_val_ratio)
    train_data = df_with_random_neg[:split_point]
    train_data = process_traindata_only(train_data) # selecting only the top 50 company names
    validation_data = df_with_random_neg[split_point:]
    
    # filtering validation and test data 
    train_users = set(train_data['jobseeker_uid'])
    validation_data = validation_data[validation_data['jobseeker_uid'].isin(train_users)]
    test_data = df_test[df_test['jobseeker_uid'].isin(train_users)]
    
    train_data = train_data.reset_index(drop=True)
    validation_data = validation_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    logger.info(f"Shape of train dataset is: {train_data.shape}")
    logger.info(f"Distribution of train dataset is: {train_data.target.value_counts()}")
    logger.info(f"Shape of validation dataset is: {validation_data.shape}")
    logger.info(f"Distribution of validation dataset is: {validation_data.target.value_counts()}")
    logger.info(f"Shape of test dataset is: {test_data.shape}")
    logger.info(f"Distribution of test dataset is: {test_data.target.value_counts()}")
    
    logger.info(f"Transforming train, and validation dataset")    
    # perform preprocessing and fit transform on train_data
    X_train = pd.DataFrame(preprocessor.fit_transform(train_data, train_data['target_w2v']))
    combined_train = pd.concat([train_data['target'], 
                                X_train, 
                                train_data[additonal_cols]], 
                               axis=1, ignore_index=True)

    # perform preprocessing and transform on validation_data
    X_val = pd.DataFrame(preprocessor.transform(validation_data))
    # Concatenate X_train and y_train
    combined_val = pd.concat([validation_data['target'], 
                              X_val, 
                              validation_data[additonal_cols]], 
                             axis=1, ignore_index=True)

    # Save processed datasets to the local paths in the processing container.
    # SageMaker will upload the contents of these paths to S3 bucket
    logger.debug("Writing processed datasets to container local path.")
    
    train_output_path = os.path.join(f"{local_dir}/train", "train.csv")   
    validation_output_path = os.path.join(f"{local_dir}/val", "validation.csv")  
    test_output_path = os.path.join(f"{local_dir}/test", "test.parquet")
    
    preprocess_pipeline_joblib_path = os.path.join(f"{local_dir}/preprocess_pipeline", "preprocess_pipeline.joblib")
    preprocess_pipeline_tar_path = os.path.join(f"{local_dir}/preprocess_pipeline", "preprocess_pipeline.tar.gz")
    
    logger.info("Saving preprocessor to {}".format(preprocess_pipeline_tar_path))
    joblib.dump(preprocessor, preprocess_pipeline_joblib_path)
    with tarfile.open(preprocess_pipeline_tar_path, "w:gz") as tar:
        tar.add(preprocess_pipeline_joblib_path,
                arcname=os.path.basename(preprocess_pipeline_joblib_path))
        
    logger.info("Saving train data to {}".format(train_output_path))
    combined_train.to_csv(train_output_path, index=False)
    
    logger.info("Saving validation data to {}".format(validation_output_path))
    combined_val.to_csv(validation_output_path, index=False)

    logger.info("Saving test data to {}".format(test_output_path))
    test_data.to_parquet(test_output_path, index=False)


def input_fn(input_data, content_type):
    try:
        if content_type == "text/csv":
            return pd.read_csv(StringIO(input_data), header=0)
        else:
            raise ValueError(f"Input fn: {content_type} not supported by script!")
    except Exception as e:
        logger.error(f"Input fn: An error occurred: {e} and the data is {input_data}")


def output_fn(prediction, accept):
    try:
        if accept == "text/csv":
            return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
        else:
            raise RuntimeError(f"Output fn: {accept} accept type is not supported by this script.")
    except Exception as e:
        logger.error(f"Output fn: An error occurred: {e} and the data is: {pd.DataFrame(prediction).to_csv(index=False)}")


def predict_fn(input_data, model):
    try:
        list_columns = ['common_working_hours', 
                        'common_employment_types', 
                        'jobseeker_preference_employment_types', 
                        'jobseeker_preference_working_hours']
        for col in list_columns:
            input_data[col] = input_data[col].str.replace("'", "\"").map(json.loads)
        input_data.loc[:,['job_allows_easy_apply', 'job_open_for_career_changers']] = input_data.loc[:,['job_allows_easy_apply', 'job_open_for_career_changers']].astype(float)
        return np.concatenate([model.transform(input_data), input_data[additonal_cols].to_numpy()], axis=1)
    except Exception as e:
        logger.error(f"Predict fn: An error occurred: {e} and the data is: {input_data.to_csv(index=False)}")
    

def model_fn(model_dir):
    try:
        preprocess_pipeline = joblib.load(os.path.join(model_dir, "preprocess_pipeline.joblib"))
        return preprocess_pipeline
    except Exception as e:
        logger.error(f"Model fn: An error occurred while loading the preprocess_pipeline: {e}")
