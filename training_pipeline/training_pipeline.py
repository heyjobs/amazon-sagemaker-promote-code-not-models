# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# ruff: noqa: E501

"""Creates and runs Sagemaker Training Pipeline"""
import json
import os
import argparse
from datetime import datetime

from aws_profiles import UserProfiles



import pandas as pd
import json
import boto3
import pathlib
import io
import sagemaker
import time

from sagemaker.model import Model
from sagemaker.sklearn import SKLearn, SKLearnModel
from sagemaker.xgboost import XGBoostModel
from sagemaker import PipelineModel

from sagemaker.deserializers import CSVDeserializer
from sagemaker.serializers import CSVSerializer

from sagemaker.xgboost.estimator import XGBoost
from sagemaker.estimator import Estimator
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import (
    ProcessingInput, 
    ProcessingOutput, 
    ScriptProcessor
)
from sagemaker.inputs import TrainingInput

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.functions import Join

from sagemaker.workflow.steps import (
    ProcessingStep, 
    TrainingStep, 
    CreateModelStep,
    CacheConfig
)
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.parameters import (
    ParameterInteger, 
    ParameterFloat, 
    ParameterString, 
    ParameterBoolean
)
from sagemaker.workflow.clarify_check_step import (
    ModelBiasCheckConfig, 
    ClarifyCheckStep, 
    ModelExplainabilityCheckConfig
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)
from sagemaker.lambda_helper import Lambda

from sagemaker.model_metrics import (
    MetricsSource, 
    ModelMetrics, 
    FileSource
)
from sagemaker.drift_check_baselines import DriftCheckBaselines

from sagemaker.image_uris import retrieve



def get_pipeline(pipeline_name: str, profile_name: str, region: str) -> Pipeline:
    sess = (
        boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
    )
    iam = sess.client("iam")
    
    # Fetch SageMaker execution role
    # sagemaker_role = sagemaker.get_execution_role()
    account_id = sess.client("sts").get_caller_identity().get("Account")
    sagemaker_role = iam.get_role(RoleName=f"{account_id}-sagemaker-exec")["Role"]["Arn"]
    
    
    # sagemaker_session = PipelineSession(boto_session=session)
    # account_id = session.client("sts").get_caller_identity().get("Account")

    # iam = session.client("iam")
    # role = iam.get_role(RoleName=f"{account_id}-sagemaker-exec")["Role"]["Arn"]

    # default_bucket = sagemaker_session.default_bucket()

    # Docker images are located in ECR in 'operations' account
    # operations_id = UserProfiles().get_profile_id("operations")
    # custom_image_uri = (
    #     f"{operations_id}.dkr.ecr.{region}.amazonaws.com/training-image:latest"
    # )

    # Set names of pipeline objects
    pipeline_name = "Job-Personalization-Pipeline"

    # for xgboost
    pipeline_model_name = "job-personalization-contextual-xgb"
    model_package_group_name = "job-personalization-contextual-xgb-group"
    base_job_name_prefix = "job-personalization"
    endpoint_config_name = f"{pipeline_model_name}-endpoint-config"
    endpoint_name = 'job-personalization-model-contextual-xgb'

    # Set data parameters
    target_col = "target"

    # important default training parameters (in sprint format)
    train_validation_ratio = "0.95"
    test_day_span = "3"
    negative_sample_ratio = "2"
    random_negative_sample_multiplier = "2"
    include_jobseeker_uid_in_training = "no"
    include_jdp_view_as_target = "no"


    # Set instance types and counts
    process_instance_type = "ml.t3.2xlarge"
    train_instance_count = 1
    train_instance_type = "ml.m5.4xlarge"

    # enable caching
    cache_config = CacheConfig(enable_caching=False, expire_after="PT5H")


    # ======================================================
    # Define Pipeline Parameters
    # ======================================================

    # Set up pipeline input parameters

    # Set timestamp param
    timestamp_param = ParameterString(
        name="timestamp", default_value="2024-03-11T11:11:11Z"
    )

    # Set timestamp param
    bucket_param = ParameterString(
        name="bucket", default_value="heyjobs-job-recommendations-production"
    )

    # Setup important parameters
    train_validation_ratio_param = ParameterString(
        name="train_validation_ratio",
        default_value=train_validation_ratio,
    )

    test_day_span_param = ParameterString(
        name="test_day_span",
        default_value=test_day_span,
    )

    negative_sample_ratio_param = ParameterString(
        name="negative_sample_ratio",
        default_value=negative_sample_ratio,
    )

    random_negative_sample_multiplier_param = ParameterString(
        name="random_negative_sample_multiplier",
        default_value=random_negative_sample_multiplier,
    )

    include_jobseeker_uid_in_training_param = ParameterString(
        name="include_jobseeker_uid_in_training",
        default_value=include_jobseeker_uid_in_training,
    )

    include_jdp_view_as_target_param = ParameterString(
        name="include_jdp_view_as_target",
        default_value=include_jdp_view_as_target,
    )

    # Set processing instance type
    process_instance_type_param = ParameterString(
        name="ProcessingInstanceType",
        default_value=process_instance_type,
    )

    # Set training instance type
    train_instance_type_param = ParameterString(
        name="TrainingInstanceType",
        default_value=train_instance_type,
    )

    # Set training instance count
    train_instance_count_param = ParameterInteger(
        name="TrainingInstanceCount",
        default_value=train_instance_count
    )

    # Set model approval param
    model_approval_status_param = ParameterString(
        name="ModelApprovalStatus", default_value="Approved"
    )

    # Set model deployment param
    model_deployment_param = ParameterString(
        name="ModelDeploymentlStatus", default_value="no"
    )

    # ======================================================
    # Step 1: Load and preprocess the data
    # ======================================================

    # Define the SKLearnProcessor configuration
    sklearn_processor = SKLearnProcessor(
        framework_version="1.0-1",
        role=sagemaker_role,
        instance_count=1,
        instance_type=process_instance_type,
        base_job_name=f"{base_job_name_prefix}-processing",
    )

    # Define pipeline processing step
    process_step = ProcessingStep(
        name="DataProcessing",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source='personalization/src/utils/', destination="/opt/ml/processing/input/code/utils/")
        ],
        outputs=[
            ProcessingOutput(
                destination=Join(on='/', values=["s3:/", bucket_param, "job-personalization/processing_jobs", timestamp_param, "train_data"]),
                output_name="train_data",
                source="/opt/ml/processing/train"
            ),
            ProcessingOutput(
                destination=Join(on='/', values=["s3:/", bucket_param, "job-personalization/processing_jobs", timestamp_param, "validation_data"]),
                output_name="validation_data",
                source="/opt/ml/processing/val")
            ,
            ProcessingOutput(
                destination=Join(on='/', values=["s3:/", bucket_param, "job-personalization/processing_jobs", timestamp_param, "test_data"]),
                output_name="test_data",
                source="/opt/ml/processing/test"
            ),
            ProcessingOutput(
                destination=Join(on='/', values=["s3:/", bucket_param, "job-personalization/processing_jobs", timestamp_param, "preprocess_pipeline"]),
                output_name="preprocess_pipeline",
                source="/opt/ml/processing/preprocess_pipeline"
            )
        ],
        job_arguments=[
            "--train-validation-ratio", train_validation_ratio_param, 
            "--test-day-span", test_day_span_param,
            "--negative-sample-ratio", negative_sample_ratio_param,
            "--random-negative-sample-multiplier", random_negative_sample_multiplier_param,
            "--include-jobseeker-uid-in-training", include_jobseeker_uid_in_training_param,
            "--include-jdp-view-as-target", include_jdp_view_as_target_param
        ],
        code="personalization/src/sk_preprocess-encoder_new.py",
        cache_config=cache_config
    )
    # ======================================================
    # Step 2: Train model
    # ======================================================

    # Retrieve training image
    xgboost_training_image = retrieve(framework="xgboost", 
                                    region=region, 
                                    version="1.7-1",
                                    image_scope="training")

    # Set XGBoost model hyperparameters 
    hyperparams = {  
            "eval_metric" : "auc",
            "objective": "binary:logistic",
            "num_round": "250",
            "max_depth": "15",
            "subsample": "0.9",
            "colsample_bytree": "0.9",
            "eta": "0.1"
            }

    estimator_output_uri = Join(on='/', values=["s3:/", bucket_param, "job-personalization/training_jobs", timestamp_param])

    xgb_estimator = Estimator(
        image_uri=xgboost_training_image,
        instance_type=train_instance_type,
        instance_count=train_instance_count,
        output_path=estimator_output_uri,
        code_location=estimator_output_uri,
        role=sagemaker_role,
    )
    xgb_estimator.set_hyperparameters(**hyperparams)

    # Access the location where the preceding processing step saved train and validation datasets
    # Pipeline step properties can give access to outputs which can be used in succeeding steps
    s3_input_train = TrainingInput(
            s3_data=process_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri, 
            content_type="text/csv"
        )
    s3_input_validation = TrainingInput(
            s3_data=process_step.properties.ProcessingOutputConfig.Outputs["validation_data"].S3Output.S3Uri,
            content_type="text/csv"
        )

    # Set pipeline training step
    train_step = TrainingStep(
            name="XGBModelTraining",
            estimator=xgb_estimator,
            inputs={
                "train":s3_input_train, # Train channel 
                "validation": s3_input_validation # Validation channel
                },
            cache_config=cache_config
            )

    # ======================================================
    # Step 3: Building sklearn pipeline model
    # ======================================================

    scaler_model = SKLearnModel(
        name="SKLearnPipelineModelXGB",
        model_data=Join(on='/', values=["s3:/", bucket_param, "job-personalization/processing_jobs", timestamp_param, "preprocess_pipeline", "preprocess_pipeline.tar.gz"]),
        role=sagemaker_role,
        sagemaker_session=sess,
        source_dir="personalization/src/",
        entry_point="sk_preprocess-encoder_new.py",
        framework_version="1.0-1",
    ) 

    scaler_model.env = {"SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT":"text/csv"}

    xgboost_inference_image = retrieve(framework="xgboost", 
                                    region=region, 
                                    version="1.7-1",
                                    image_scope="inference")

    model_artifacts = train_step.properties.ModelArtifacts.S3ModelArtifacts

    xgboost_model =  Model(
        image_uri=xgboost_inference_image,
        model_data=model_artifacts,
        sagemaker_session=sess,
        role=sagemaker_role,
    )

    pipeline_model = PipelineModel(
        models=[scaler_model, xgboost_model], role=sagemaker_role, sagemaker_session=sess
    )

    # ======================================================
    # Step 4: Evaluate model
    # ======================================================

    eval_processor = ScriptProcessor(
        image_uri=xgboost_training_image,
        command=["python3"],
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name=f"{base_job_name_prefix}-model-eval",
        sagemaker_session=sess,
        role=sagemaker_role,
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    # Set model evaluation step
    evaluation_step = ProcessingStep(
        name="XGBModelEvaluate",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                # Fetch S3 location where train step saved model artifacts
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source='personalization/src/utils/', 
                destination="/opt/ml/processing/input/code/utils/",
            ),
            ProcessingInput(
                source=process_step.properties.ProcessingOutputConfig.Outputs["preprocess_pipeline"].S3Output.S3Uri,
                destination="/opt/ml/processing/preprocess_pipeline",
            ),
            ProcessingInput(
                # Fetch S3 location where processing step saved test data
                source=process_step.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(destination=Join(on='/', values=["s3:/", bucket_param, "job-personalization/model_eval", timestamp_param]),
                            output_name="evaluation", 
                            source="/opt/ml/processing/evaluation"),
        ],
        job_arguments=[
            "--include-jobseeker-uid-in-training", include_jobseeker_uid_in_training_param
        ],
        code="personalization/src/xgboost_evaluate-encoder.py",
        property_files=[evaluation_report],
        cache_config=cache_config
    )

    # ======================================================
    # Step 5: Register model
    # ======================================================

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(on='/', values=["s3:/", bucket_param, "job-personalization/model_eval", timestamp_param, "evaluation.json"]),
            content_type="application/json",
        )
    )

    register_step = RegisterModel(
        name="XGBRegisterModel",
        model=pipeline_model,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.t2.large", "ml.t2.xlarge", "ml.m5.large", "ml.m6g.large"],
        model_metrics=model_metrics,
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status_param,
    )
    pipeline_model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.t2.large", "ml.t2.xlarge", "ml.m5.large", "ml.m6g.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status_param, 
    )

    # ======================================================
    # Step 6: Condition for model approval status
    # ======================================================

    # Evaluate model performance on test set
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="classification_metrics.roc_auc.value",
        ),
        right=0.6, # Threshold to compare model performance against
    )

    condition_step = ConditionStep(
        name="CheckPersonalizationModelXGBEvaluation",
        conditions=[cond_gte],
        if_steps=[register_step], 
        else_steps=[]
    )

    # ======================================================
    # Final Step: Define Pipeline
    # ======================================================

    # Create the Pipeline with all component steps and parameters
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[timestamp_param,
                    bucket_param,
                    process_instance_type_param, 
                    train_instance_type_param, 
                    train_instance_count_param, 
                    model_approval_status_param,
                    train_validation_ratio_param,
                    test_day_span_param,
                    negative_sample_ratio_param,
                    random_negative_sample_multiplier_param,
                    include_jobseeker_uid_in_training_param,
                    include_jdp_view_as_target_param],
        steps=[
            process_step,
            train_step,
            evaluation_step,
            condition_step
        ],
        sagemaker_session=sess
    )
    return pipeline


def create_pipeline(pipeline_name, profile, region):
    """Create/update pipeline"""
    pipeline = get_pipeline(
        pipeline_name=pipeline_name,
        profile_name=profile,
        region=region,
    )
    json.loads(pipeline.definition())

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    account_id = session.client("sts").get_caller_identity().get("Account")
    iam = session.client("iam")
    role = iam.get_role(RoleName=f"{account_id}-sagemaker-exec")["Role"]["Arn"]
    pipeline.upsert(role_arn=role)


def run_pipeline(pipeline_name: str, profile_name: str = None) -> None:
    session = (
        boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
    )
    sagemaker_client = session.client("sagemaker")
    sagemaker_client.start_pipeline_execution(PipelineName=pipeline_name)


if __name__ == "__main__":
    userProfiles = UserProfiles()
    profiles = userProfiles.list_profiles()

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=None, choices=profiles)
    parser.add_argument("--region", type=str, default="eu-west-3")
    parser.add_argument("--pipeline-name", type=str, default="training-pipeline")
    parser.add_argument("--action", type=str, choices=["create", "run"])
    args = parser.parse_args()

    if args.action == "create":
        create_pipeline(args.pipeline_name, args.profile, args.region)

    elif args.action == "run":
        run_pipeline(args.pipeline_name, args.profile)
