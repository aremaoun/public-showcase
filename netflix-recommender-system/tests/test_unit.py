"""A unit test."""

import pathlib

from netflix_recommender_system.preprocessing import Preprocessing
import pandas as pd


def test_compute_features(config, teardown_resource_directory, snapshot):
    """Test compute features method."""
    preprocessing_pipeline = Preprocessing(
        mode="training",
        input_datafilepath=pathlib.Path(
            config["DATA_DIRECTORY_PATH"], "data_sample.csv"
        ),
        output_datafilepath="./tests/resources/training_dataset_features.csv",
    )

    df = preprocessing_pipeline.compute_features()
    column_features = ",".join(list(df.columns))

    assert isinstance(df, pd.DataFrame)

    snapshot.assert_match(column_features, snapshot_name="column_features")
