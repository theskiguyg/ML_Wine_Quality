import pandas as pd
import numpy as np
from preprocess import get_new_preprocessor
from utils import load_config, save_model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import r2_score

def start_train(config_path):
    config = load_config(config_path)
    mode = config['mode']

    X_train = pd.read_csv(config['data']['X_train_path'])
    y_train = pd.read_csv(config['data']['y_train_path'])
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

    preprocessor = get_new_preprocessor(config)
    final_pipeline = Pipeline([
        ('preprocessing_pipeline', preprocessor), 
        ('model', model)
        ])
    
    if mode == "train":
        print("Train mode is selected.")
        final_pipeline.fit(X_train, y_train)
        print(f"Model {model_name} has been trained.")
        predictions = final_pipeline.predict(X_train)
        scoring = r2_score(y_train, predictions)
        print(f"R2 scoring: {scoring}")
    elif mode == 'cross_val':
        print("CrossValidation mode is selected.")
        kfold = KFold(**config['cross_val']['KFold'])
        cv_scores = cross_val_score(final_pipeline,
                                X_train, y_train, 
                                cv=kfold, 
                                scoring=config['cross_val']['scoring'])
        print("CrossValidation is done.")
        print(f"Scores: {cv_scores}")
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
        print(f"Best score: {best_score}")
        print(f"Best_params: {best_params}")
    elif mode == "save":
        print("Save mode is selected.")
        final_pipeline.fit(X_train, y_train)
        print("Model has been trained.")
        save_model(final_pipeline, config['model_save_path'])
        print(f"Model is saved to {config['model_save_path']}")

if __name__ == '__main__':
    start_train('configs/baseline.yaml')