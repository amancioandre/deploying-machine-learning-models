from sklearn.pipeline import Pipeline

import preprocessor as pp

CATEGORICAL_VARS = [
    'MSZoning',
    'Neighborhood',
    'RoofStyle',
    'MasVnrType',
    'BsmtQual',
    'BsmtExposure',
    'HeatingQC',
    'CentralAir',
    'KitchenQual',
    'FireplaceQu',
    'GarageType',
    'GarageFinish',
    'PavedDrive'
]

PIPELINE_NAME = 'lasso_regression'

price_pipe = Pipeline([
    ('categorigal_imputer', pp.CategoricalImputer(variables=CATEGORICAL_VARS))
])
