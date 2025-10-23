import sys
import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # ADD StandardScaler import
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


from src.constants.training_pipeline import TARGET_COLUMN
from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.exception.exception import DeliveryTimeException
from src.logging.logger import logging
from src.utils.main_utils.utils import save_numpy_array_data, save_object


class NullSafeOneHotEncoder(BaseEstimator, TransformerMixin):
    """Custom transformer that applies OneHotEncoder only to non-null values"""


    def __init__(self, drop='first', handle_unknown='ignore'):
        self.drop = drop
        self.handle_unknown = handle_unknown
        self.encoder = None
        self.columns = None


    def fit(self, X, y=None):


        if hasattr(X, 'columns'):
            self.columns = X.columns.tolist()
        else:
            self.columns = [f'col_{i}' for i in range(X.shape[1])]



        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)



        self.encoder = OneHotEncoder(drop=self.drop, handle_unknown=self.handle_unknown, sparse_output=False)



        self.fitted_columns = {}
        for col in self.columns:
            non_null_mask = X[col].notna()
            if non_null_mask.any():
                non_null_data = X.loc[non_null_mask, [col]]
                if col not in self.fitted_columns:


                    col_encoder = OneHotEncoder(drop=self.drop, handle_unknown=self.handle_unknown, sparse_output=False)
                    col_encoder.fit(non_null_data)
                    self.fitted_columns[col] = col_encoder


        return self


    def transform(self, X):


        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)


        result_dfs = []


        for col in self.columns:
            if col in self.fitted_columns:
                encoder = self.fitted_columns[col]



                col_data = X[[col]].copy()


                feature_names = encoder.get_feature_names_out([col])
                encoded_result = pd.DataFrame(
                    np.zeros((len(X), len(feature_names))),
                    columns=feature_names,
                    index=X.index
                )



                non_null_mask = col_data[col].notna()
                if non_null_mask.any():
                    non_null_data = col_data.loc[non_null_mask]
                    transformed_data = encoder.transform(non_null_data)



                    encoded_result.loc[non_null_mask] = transformed_data



                null_mask = col_data[col].isna()
                encoded_result.loc[null_mask] = np.nan


                result_dfs.append(encoded_result)
            else:
                # If column wasn't fitted (all nulls), create dummy columns filled with NaN
                dummy_cols = pd.DataFrame(
                    np.full((len(X), 1), np.nan),
                    columns=[f'{col}_dummy'],
                    index=X.index
                )
                result_dfs.append(dummy_cols)


        # Concatenate all encoded columns
        if result_dfs:
            return pd.concat(result_dfs, axis=1)
        else:
            return pd.DataFrame(index=X.index)


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise DeliveryTimeException(e, sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise DeliveryTimeException(e, sys)


    def preprocess_data(self, df):
        try:
            df = df.drop(['Order_ID'], axis=1)
            df.loc[df['Vehicle_Type'] == 'Car', 'Vehicle_Type'] = 'Pickup Truck'
            df['Actual_Delivery_Time'] = df['Delivery_Time_min'] - df['Preparation_Time_min']
            df=df.drop(['Preparation_Time_min', 'Delivery_Time_min'], axis=1)


            return df
        except Exception as e:
            raise DeliveryTimeException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Starting data transformation")
        try:
            # Load train/test
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df  = self.read_data(self.data_validation_artifact.valid_test_file_path)


            train_df = self.preprocess_data(train_df)
            test_df  = self.preprocess_data(test_df)


            # Separate features/target
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]


            X_test  = test_df.drop(columns=[TARGET_COLUMN])
            y_test  = test_df[TARGET_COLUMN]


            # CHANGED: Define pipeline with StandardScaler for numerical features
            full_pipeline = Pipeline([
                ("encoders", ColumnTransformer([
                    ("scaler", StandardScaler(), ["Distance_km", "Courier_Experience_yrs"]),  # ADD THIS LINE
                    ("veh", NullSafeOneHotEncoder(drop="first"), ["Vehicle_Type"]),
                    ("weather", NullSafeOneHotEncoder(drop="first"), ["Weather"]),
                    ("time", NullSafeOneHotEncoder(drop="first"), ["Time_of_Day"]),
                    ("traffic", NullSafeOneHotEncoder(drop="first"), ["Traffic_Level"]),
                ], remainder="passthrough")),
                ("imputer", IterativeImputer(estimator=RandomForestRegressor(),
                                             max_iter=10, random_state=42))
            ])


            # Fit/transform train, transform test
            X_train_transformed = full_pipeline.fit_transform(X_train)
            X_test_transformed  = full_pipeline.transform(X_test)




            # Combine features and target
            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr  = np.c_[X_test_transformed, y_test.to_numpy()]


            # Save transformed arrays
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)


            # Save pipeline
            save_object(self.data_transformation_config.transformed_object_file_path, full_pipeline)


            logging.info("Data transformation completed successfully")


            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )


        except Exception as e:
            raise DeliveryTimeException(e, sys)
