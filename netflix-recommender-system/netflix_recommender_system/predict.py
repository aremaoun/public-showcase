"""Predict."""

import pathlib
import dill as pickle
import copy

from netflix_recommender_system.config import config
from netflix_recommender_system.preprocessing import Preprocessing


class SupervisedLearning(object):
    pass


def run(data: dict):
    # PREDICTION_DICT = {
    #     "customer_id": 1044034,
    #     # "rating": None,
    #     "rating_date": "2005-02-03",
    #     "movie_id": 12031,
    #     "release_year": 2002,
    #     "title": "Scotland",
    # }

    PREDICTION_DICT = data

    # return config
    # SupervisedLearning

    with open(
        pathlib.Path(
            config["MODEL_DIRECTORY_PATH"],
            str(config["MODEL_TIMESTAMP_ID"]),
            "model.pickle",
        ),
        "rb",
    ) as file:
        learning_pipeline = pickle.load(file)
    # with open("model.pickle", "rb") as file:
    #     learning_pipeline = pickle.load(file)

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
