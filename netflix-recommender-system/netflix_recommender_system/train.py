"""Training of model."""

import pathlib
import time
import dill as pickle
import yaml
import os

from netflix_recommender_system.logger import logger
from netflix_recommender_system.config import config
from netflix_recommender_system.preprocessing import Preprocessing
from netflix_recommender_system.model import SupervisedLearning


def run():
    MODEL_TIMESTAMP_ID_NEW = str(time.time()).split(".")[0]

    os.makedirs(pathlib.Path(config["MODEL_DIRECTORY_PATH"], MODEL_TIMESTAMP_ID_NEW))

    preprocessing_pipeline = Preprocessing(
        mode="training",
        input_datafilepath=pathlib.Path(
            config["DATA_DIRECTORY_PATH"], "data_sample.csv"
        ),
        output_datafilepath=pathlib.Path(
            config["MODEL_DIRECTORY_PATH"],
            MODEL_TIMESTAMP_ID_NEW,
            "training_dataset_features.csv",
        ),
    )

    df = preprocessing_pipeline.compute_features()

    learning_pipeline = SupervisedLearning(
        training_df=df[df["set"] == "training"],
        validation_df=df[df["set"] == "validation"],
        numeric_features_names=[
            "rating_year",
            "avg_rating_of_similar_customers",
            "number_of_similar_customers",
            "rating_x_era_avg",
        ],
        categorical_features_names=["title", "release_era"],
        y_name="rating",
        seed=config["SEED"],
    )

    learning_pipeline.create_training_arrays()

    learning_pipeline.create_validation_arrays()

    learning_pipeline.train()

    learning_pipeline.validate()

    logger.info(f"RMSE on training set is: {learning_pipeline.rmse_training}")
    logger.info(f"RMSE on validat set is: {learning_pipeline.rmse_validation}")

    summary = {
        "rmse_training": learning_pipeline.rmse_training,
        "rmse_validation": learning_pipeline.rmse_validation,
    }

    with open(
        pathlib.Path(
            config["MODEL_DIRECTORY_PATH"], MODEL_TIMESTAMP_ID_NEW, "model.pickle"
        ),
        "wb",
    ) as file:
        pickle.dump(learning_pipeline, file)

    with open(
        pathlib.Path(
            config["MODEL_DIRECTORY_PATH"], MODEL_TIMESTAMP_ID_NEW, "summary.yaml"
        ),
        "w",
    ) as file:
        yaml.dump(summary, file)

    logger.info("Model exported successfully")
