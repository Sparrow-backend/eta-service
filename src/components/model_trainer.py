import os
import sys


from src.exception.exception import DeliveryTimeException
from src.logging.logger import logging


from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig


from src.utils.ml_utils.model.estimator import DeliveryPredictionModel
from src.utils.main_utils.utils import save_object, load_object
from src.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from src.utils.ml_utils.metric.regression_metric import get_regression_score


import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import os
import joblib
import sys
import mlflow


from dotenv import load_dotenv
load_dotenv()


os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
            # CHANGED: Update feature names to match the pipeline output with scaler
            self.model_trainer_config.feature_names=[
                'scaler__Distance_km',
                'scaler__Courier_Experience_yrs',
                'Vehicle_Type_Pickup Truck',
                'Vehicle_Type_Scooter',
                'Weather_Foggy',
                'Weather_Rainy',
                'Weather_Snowy',
                'Weather_Windy',
                'Time_of_Day_Evening',
                'Time_of_Day_Morning',
                'Time_of_Day_Night',
                'Traffic_Level_Low',
                'Traffic_Level_Medium'
            ]
        except Exception as e:
            raise DeliveryTimeException(e, sys)


    def track_mlflow(self, best_model, regressionMetric):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))


        with mlflow.start_run():
            mlflow.log_metric("r2_score", regressionMetric.r2_score)
            mlflow.log_metric("Mean_Absolute_Error", regressionMetric.mean_absolute_error)
            mlflow.log_metric("Mean_Squared_Error", regressionMetric.mean_squared_error)


            joblib.dump(best_model, "model.joblib")


            mlflow.log_artifact("model.joblib", artifact_path="model")


    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "XGBoost Regression": XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
                "RandomForest": RandomForestRegressor(),
                "GradientBoostRegressor": GradientBoostingRegressor()
            }


            params = {
                "XGBoost Regression" : {
                    'n_estimators': [500, 700, 100, 150],
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.01, 0.05],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8]
                },
                'GradientBoostRegressor': {
                    'n_estimators':[1000, 500],
                    'min_samples_split': [2, 8],
                    'criterion': ['friedman_mse', 'squared_error'],
                    'loss': ['squared_error', 'huber'],
                    'max_depth': [5, None]
                },
                'RandomForest': {
                    'n_estimators': [1000],
                    'min_samples_split': [2],
                    'max_features': [7],
                    'max_depth': [None]
                }
            }


            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              models=models, param=params)


            best_model_score = max(sorted(model_report.values()))


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]


            best_model = models[best_model_name]


            y_train_pred = best_model.predict(X_train)


            regression_train_metric=get_regression_score(y_true= y_train, y_pred=y_train_pred)
            self.track_mlflow(best_model, regression_train_metric)


            y_test_pred = best_model.predict(X_test)
            regression_test_metric = get_regression_score(y_true=y_test, y_pred=y_test_pred)


            self.track_mlflow(best_model, regression_test_metric)


            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)


            Delivery_Prediction_Model=DeliveryPredictionModel(model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=Delivery_Prediction_Model)


            # Model Pusher
            save_object("final_model/model.pkl", best_model)


            model_trainer_artifact=ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=regression_train_metric,
                test_metric_artifact=regression_test_metric
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact


        except Exception as e:
            raise DeliveryTimeException(e, sys)


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path


            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)


            logging.info(f"Shape of training data: {train_arr.shape}")
            logging.info(f"Testing array: {test_arr.shape}")


            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )


            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"X_test sahpe: {X_test.shape}")
            
            # REMOVED: All the StandardScaler code and DataFrame conversion
            # Train directly on the transformed arrays
            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact


        except Exception as e:
            raise DeliveryTimeException(e, sys)
