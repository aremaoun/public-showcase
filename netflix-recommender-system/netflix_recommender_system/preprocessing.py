"""Preprocessing."""

import datetime

import pandas as pd


class Preprocessing:
    """Preprocess."""

    def __init__(
        self,
        mode: str,
        input_datafilepath: str,
        output_datafilepath=None,
        prediction_dict=None,
    ):
        if mode not in ["training", "prediction"]:
            raise ValueError("Mode must be one of 'training' or 'prediction'")
        self.mode = mode
        self.input_datafilepath = input_datafilepath
        self.output_datafilepath = output_datafilepath
        self.prediction_dict = prediction_dict

    def __load_data_and_clean(self) -> pd.DataFrame:
        """Load input data file."""
        df = pd.read_csv(self.input_datafilepath)
        df["rating_date"] = pd.to_datetime(df["rating_date"], format="mixed")
        df["rating_year"] = pd.DatetimeIndex(df["rating_date"]).year

        def get_set(year):
            if year <= 2003:
                return "training"
            elif year == 2004:
                return "validation"
            elif year == 2005:
                return "prediction"

        df["set"] = df["rating_year"].apply(lambda x: get_set(x))
        df = df[~(df["set"] == "prediction")]

        return df

    def __load_data_and_clean_and_insert_prediction_row(self) -> pd.DataFrame:
        """Insert prediction row."""

        df = self.__load_data_and_clean()

        toto = self.prediction_dict["rating_date"]
        self.prediction_dict["rating_date"] = datetime.datetime.strptime(
            toto, "%Y-%m-%d"
        )

        df = pd.concat(
            [
                df,
                pd.DataFrame.from_dict(
                    {
                        "customer_id": [self.prediction_dict["customer_id"]],
                        "rating_date": [self.prediction_dict["rating_date"]],
                        "movie_id": [self.prediction_dict["movie_id"]],
                        "release_year": [self.prediction_dict["release_year"]],
                        "title": [self.prediction_dict["title"]],
                        "rating": [None],
                        "rating_year": [2005],
                        "set": ["prediction"],
                    }
                ),
            ],
            ignore_index=True,
        )

        return df

    def __write_data(self, df: pd.DataFrame):
        """Write features data file."""
        df.to_csv(self.output_datafilepath, sep=",", index=False)

    def compute_features(self):
        """Compute features."""
        if self.mode == "training":
            df = self.__load_data_and_clean()
        else:
            df = self.__load_data_and_clean_and_insert_prediction_row()

        def get_era(year):
            if year < 1970:
                return "<1970"
            elif year < 1990:
                return "70s and 80s"
            else:
                return "90s and 2000s"

        df["release_era"] = df["release_year"].apply(lambda x: get_era(x))

        movie_id_list = set(df["movie_id"])

        counter = 0
        for movie_id in movie_id_list:
            ratings_import_movie_df = df[df["movie_id"] == movie_id]
            ratings_import_movie_df = ratings_import_movie_df.merge(
                ratings_import_movie_df, on=["movie_id"], how="inner"
            )
            ratings_import_movie_df = ratings_import_movie_df[
                ~(
                    ratings_import_movie_df.customer_id_x
                    == ratings_import_movie_df.customer_id_y
                )
                & ~(ratings_import_movie_df.set_y == "prediction")
            ]

            if counter == 0:
                ratings_feature_movie_df = ratings_import_movie_df
            else:
                ratings_feature_movie_df = pd.concat(
                    [ratings_feature_movie_df, ratings_import_movie_df]
                )
            counter += 1

        ratings_feature_movie_df["rating_gap"] = abs(
            ratings_feature_movie_df["rating_x"] - ratings_feature_movie_df["rating_y"]
        )

        movie_id_list = set(df["movie_id"])

        counter = 0
        for movie_id in movie_id_list:
            cross_join_df = pd.merge(
                ratings_feature_movie_df[
                    ratings_feature_movie_df["movie_id"] == movie_id
                ],
                ratings_feature_movie_df[
                    (ratings_feature_movie_df["movie_id"] != movie_id)
                    & (ratings_feature_movie_df["set_y"] == "training")
                ]
                .groupby(["customer_id_x", "customer_id_y"])["rating_gap"]
                .mean(),
                left_on=["customer_id_x", "customer_id_y"],
                right_index=True,
                how="inner",
                suffixes=["", "_avg"],
            )  # .rename(columns={"rating_gap_x": "avg_rating_gap"})
            avg_rating_of_similar_customers_movie_df = (
                cross_join_df[cross_join_df["rating_gap_avg"] <= 0.5]
                .groupby(["customer_id_x", "movie_id"])["rating_y"]
                .mean()
                .rename("avg_rating_of_similar_customers")
            )
            number_of_similar_customers_movie_df = (
                cross_join_df[cross_join_df["rating_gap_avg"] <= 0.5]
                .groupby(["customer_id_x", "movie_id"])["rating_y"]
                .size()
                .rename("number_of_similar_customers")
            )

            rating_x_era_avg_movie_df = pd.merge(
                ratings_feature_movie_df[
                    ratings_feature_movie_df["movie_id"] == movie_id
                ].drop_duplicates(["customer_id_x", "movie_id"]),
                ratings_feature_movie_df[
                    (ratings_feature_movie_df["movie_id"] != movie_id)
                    & (ratings_feature_movie_df["set_x"] == "training")
                ]
                .drop_duplicates(["customer_id_x", "movie_id"])
                .groupby(["customer_id_x", "release_era_x"])["rating_x"]
                .mean(),
                left_on=["customer_id_x", "release_era_x"],
                right_index=True,
                how="inner",
                suffixes=["", "_era_avg"],
            )[["customer_id_x", "movie_id", "rating_x_era_avg"]]

            if counter == 0:
                avg_rating_of_similar_customers_df = (
                    avg_rating_of_similar_customers_movie_df
                )
                number_of_similar_customers_df = number_of_similar_customers_movie_df
                rating_x_era_avg_df = rating_x_era_avg_movie_df
            else:
                avg_rating_of_similar_customers_df = pd.concat(
                    [
                        avg_rating_of_similar_customers_df,
                        avg_rating_of_similar_customers_movie_df,
                    ]
                )
                number_of_similar_customers_df = pd.concat(
                    [
                        number_of_similar_customers_df,
                        number_of_similar_customers_movie_df,
                    ]
                )
                rating_x_era_avg_df = pd.concat(
                    [rating_x_era_avg_df, rating_x_era_avg_movie_df]
                )
            counter += 1

        ratings_features_export_df = (
            pd.merge(
                df,
                avg_rating_of_similar_customers_df,
                left_on=["customer_id", "movie_id"],
                right_index=True,
                how="left",
            )
            .merge(
                number_of_similar_customers_df,
                left_on=["customer_id", "movie_id"],
                right_index=True,
                how="left",
            )
            .merge(
                rating_x_era_avg_df,
                left_on=["customer_id", "movie_id"],
                right_on=["customer_id_x", "movie_id"],
                how="left",
            )
            .drop(columns=["customer_id_x"])
        )

        avg_rating_of_similar_customers_imputed_value = ratings_features_export_df.loc[
            ratings_features_export_df["set"] == "training", "rating"
        ].mean(skipna=True)
        number_of_similar_customers_imputed_value = (
            0  # technically not an imputed value
        )
        rating_x_era_avg_imputed_value = ratings_features_export_df.loc[
            ratings_features_export_df["set"] == "training", "rating"
        ].mean(
            skipna=True
        )  # quick and dirty (would be better to have it by group)

        ratings_features_export_df.loc[
            ratings_features_export_df["avg_rating_of_similar_customers"].isna(),
            "avg_rating_of_similar_customers",
        ] = avg_rating_of_similar_customers_imputed_value
        ratings_features_export_df.loc[
            ratings_features_export_df["number_of_similar_customers"].isna(),
            "number_of_similar_customers",
        ] = number_of_similar_customers_imputed_value
        ratings_features_export_df.loc[
            ratings_features_export_df["rating_x_era_avg"].isna(), "rating_x_era_avg"
        ] = rating_x_era_avg_imputed_value

        if self.mode == "training":
            self.__write_data(df=ratings_features_export_df)
        else:
            ratings_features_export_df = ratings_features_export_df[
                ratings_features_export_df["set"] == "prediction"
            ]

        return ratings_features_export_df
