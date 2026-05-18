from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
import numpy as np

def get_preprocessor(config):
    """ Preprocessor for EDA and comparison scaled data with stock data"""
    scale_cols = config['preprocess'].get('scale_cols', [])
    log_cols = config['preprocess'].get('log_cols', [])
    
    transformers = []
    
    if scale_cols:
        transformers.append(('scale', StandardScaler(), scale_cols))
    
    if log_cols:
        log_pipeline = Pipeline([
            ('log', FunctionTransformer(np.log1p, validate=True)),
            ('scale', StandardScaler())
        ])
        transformers.append(('log', log_pipeline, log_cols))
    
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    return preprocessor
