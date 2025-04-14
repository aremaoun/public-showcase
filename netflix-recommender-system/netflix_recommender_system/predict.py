"""Predict."""

import pathlib
import copy

import dill as pickle

from netflix_recommender_system.config import config
from netflix_recommender_system.preprocessing import Preprocessing


def run(data: dict):

    PREDICTION_DICT = data

    with open(
        pathlib.Path(
            config["MODEL_DIRECTORY_PATH"],
            str(config["MODEL_TIMESTAMP_ID"]),
            "model.pickle",
        ),
        "rb",
    ) as file:
        learning_pipeline = pickle.load(file)

    preprocessing_pipeline = Preprocessing(
        mode="prediction",
        input_datafilepath=str(
            pathlib.Path(config["DATA_DIRECTORY_PATH"], "data_sample.csv")
        ),
        prediction_dict=copy.deepcopy(
            PREDICTION_DICT
        ),  # deep copy to prevent side-effect change of dict argument in func
    )

    return learning_pipeline.predict(
        prediction_df=preprocessing_pipeline.compute_features()
    )[0].item()
