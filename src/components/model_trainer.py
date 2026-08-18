import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'absolute_error', 'poisson'],
                    'max_features':['sqrt','log2',None],
                    'max_depth': [3,5,7,10]
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256],
                    'max_features': ['sqrt','log2']
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.001,.05,.005,.0001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbors Regressor":{
                    'n_neighbors': [5,7,9,11],
                    'weights': ['uniform','distance'],
                    'algorithm': ['ball_tree','kd_tree','brute']
                },
                "XGBRegressor":{
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate': [.1,.01,.001,.0001],
                    'max_depth': [3,5,7,9]
                },
                "CatBoosting Regressor":{
                    'iterations': [8,16,32,64,128,256],
                    'learning_rate': [.1,.01,.001,.0001],
                    'depth': [3,5,7,9]
                },
                "AdaBoost Regressor":{
                    'learning_rate': [.1,.01,.001,.0001, .00001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found", sys)
            logging.info(f"Best model found on both training and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                )

            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)

    trainer = ModelTrainer()
    print(trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path))
