"""Fixtures."""

import yaml
import os
import pytest


@pytest.fixture(scope="session")
def teardown_resource_directory():
    """Fixture to create resources directory before the test and remove it after."""
    try:
        os.remove("tests/resources/training_dataset_features.csv")
    except OSError:
        pass
    try:
        os.makedirs("tests/resources")
    except OSError:
        pass
    yield
    try:
        os.remove("tests/resources/training_dataset_features.csv")
    except OSError:
        pass
    try:
        os.rmdir("tests/resources")
    except OSError:
        pass


@pytest.fixture
def config():
    """Foo."""

    with open("./config.yaml") as file:
        config = yaml.safe_load(file)

    return config
