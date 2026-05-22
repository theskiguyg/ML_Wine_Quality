import pandas as pd
import numpy as np
from preprocess import get_new_preprocessor
from utils import load_config, save_model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import r2_score
import mlflow
def start_train(config_path):
    config = load_config(config_path)
    mode = config['mode']

    X_train = pd.read_csv(config['data']['X_train_path'])
    y_train = pd.read_csv(config['data']['y_train_path']).values.ravel()
    X_train = X_train.drop('Id', axis=1)

    model_name = config['using_model']

    if model_name == "LinearRegression":
        model = LinearRegression(**config['models'].get('LinearRegression', {}))
    elif model_name == "RandomForestRegressor":
        model = RandomForestRegressor(**config['models'].get('RandomForestRegressor', {}))
    elif model_name == "Ridge":
        model = Ridge(**config['models'].get('Ridge', {}))
    elif model_name == "Lasso":
        model = Lasso(**config['models'].get('Lasso', {}))
    elif model_name == "GradientBoostingRegressor":
        model = GradientBoostingRegressor(**config['models'].get('GradientBoostingRegressor', {}))
    else: 
        raise ValueError("Unknown model, change config.")

    preprocessor = get_new_preprocessor(config)
    final_pipeline = Pipeline([
        ('preprocessing_pipeline', preprocessor), 
        ('model', model)
        ])
    
    mlflow.set_experiment(f"Wine_Quality_{mode}")
    
    with mlflow.start_run(run_name=f"{model_name}_{mode}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("mode", mode)
        mlflow.log_params(config['models'].get(model_name, {}))
    
        if mode == "train":
            print("Train mode is selected.")
            final_pipeline.fit(X_train, y_train)
            print(f"Model {model_name} has been trained.")
            predictions = final_pipeline.predict(X_train)
            scoring = r2_score(y_train, predictions)
            print('Model is not saved!')
            mlflow.log_metric("train_r2", scoring)
            print(f"MLflow run_id: {mlflow.active_run().info.run_id}")
            print(f"Train R²: {scoring:.4f}")

        elif mode == 'cross_val':
            print("CrossValidation mode is selected.")
            kfold = KFold(**config['cross_val']['KFold'])

            cv_scores = cross_val_score(final_pipeline,
                                    X_train, y_train, 
                                    cv=kfold, 
                                    scoring=config['cross_val']['scoring'])
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())
            mlflow.log_params(config['cross_val']['KFold'])
            print("CrossValidation is done.")
            print(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"CV scores: {cv_scores}")
            print(f"MLflow run_id: {mlflow.active_run().info.run_id}")
        elif mode == "grid_search":
            param_grid = config['grid_search']['param_grid']
            grid_search = GridSearchCV(final_pipeline,
                                    param_grid=param_grid, 
                                    cv=config['grid_search']['cv'],
                                    scoring=config['grid_search']['scoring']
                                    )
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            mlflow.log_metric("best_score", grid_search.best_score_)
            mlflow.log_params(grid_search.best_params_)
            print(f"Best score: {grid_search.best_score_:.4f}")
            print(f"Best params: {grid_search.best_params_}")
            print(f"MLflow run_id: {mlflow.active_run().info.run_id}")

        elif mode == "save":
            print("Save mode is selected.")
            final_pipeline.fit(X_train, y_train)
            predictions = final_pipeline.predict(X_train)
            scoring = r2_score(y_train, predictions)
            mlflow.log_metric("train_r2", scoring)
            print("Model has been trained.")
            save_model(final_pipeline, config['model_save_path'])
            print(f"Model saved to {config['model_save_path']}")
            print(f"Train R²: {scoring:.4f}")

if __name__ == '__main__':
    start_train('configs/baseline.yaml')