"""Supervised learning action definitions."""
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import pandas as pd


class SupervisedLearning:
    """Actions for supervised learning tasks."""

    def __init__(self,
                 training_df: pd.Dataframe,
                 validation_df: pd.Dataframe,
                 numeric_features_names: list[str],
                 categorical_features_names: list[str],
                 y_name: str,
                 seed: int = 0):
        """Instantiate object."""

        self.training_df = training_df
        self.validation_df = validation_df
        self.numeric_features_names = numeric_features_names
        self.categorical_features_names = categorical_features_names
        self.y_name = y_name
        self.seed = seed

    def create_training_arrays(self):
        """Create training arrays."""
        
        self.numeric_features_training = self.training_df[self.numeric_features_names].to_numpy()
        self.categorical_features_training = self.training_df[self.categorical_features_names].to_numpy()
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.encoder.fit(self.categorical_features_training)
        self.encoded_categorical_features_training = self.encoder.transform(self.categorical_features_training).toarray()
        self.X_training = np.concatenate((self.numeric_features_training, self.encoded_categorical_features_training), axis = 1)
        self.y_training = self.training_df[self.y_name].to_numpy()
    
    def create_validation_arrays(self):
        """Create validation arrays."""

        self.numeric_features_validation = self.validation_df[self.numeric_features_names].to_numpy()
        self.categorical_features_validation = self.validation_df[self.categorical_features_names].to_numpy()
        self.encoded_categorical_features_validation = self.encoder.transform(self.categorical_features_validation).toarray()
        self.X_validation = np.concatenate((self.numeric_features_validation, self.encoded_categorical_features_validation), axis = 1)
        self.y_validation = self.validation_df[self.y_name].to_numpy()
    
    def train(self):
        """Train model."""
        
        self.model = RandomForestRegressor(max_depth=1, max_features = 2, random_state=self.seed)
        self.model.fit(self.X_training, self.y_training)

    def validate(self):
        """Pass."""
        pass

    def predict(self, prediction_dict: dict):
        """Pass."""

        prediction_df = pd.DataFrame([prediction_dict])
        self.numeric_features_prediction = prediction_df[self.numeric_features_names].to_numpy()
        self.categorical_features_prediction = prediction_df[self.categorical_features_names].to_numpy()
        self.encoded_categorical_features_prediction = self.encoder.transform(self.categorical_features_prediction).toarray()
        self.X_prediction = np.concatenate((self.numeric_features_prediction, self.encoded_categorical_features_prediction), axis = 1)
        print(self.model.predict(self.X_prediction))
