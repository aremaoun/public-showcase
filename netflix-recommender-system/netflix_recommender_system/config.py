"""Load configuration file."""

import yaml


with open("./config.yaml") as file:
    config = yaml.safe_load(file)
