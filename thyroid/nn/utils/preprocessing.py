from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

class AgeOutlierImputer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle age outliers by imputing values over 150 
    with the median age of the dataset.
    """
    def __init__(self, threshold=150):
        self.threshold = threshold
        self.median_age = None
        
    def fit(self, X, y=None):
        # Calculate median age excluding outliers
        self.median_age = np.median(X[X <= self.threshold])
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        mask = X_copy > self.threshold
        X_copy[mask] = self.median_age
        return X_copy


class TSHLogTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle TSH's right-skewed distribution 
    using log transformation with handling of zeros.
    """
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        # Add small constant before log transform to handle zeros
        return np.log1p(X_copy)


standard_numeric = ['TT4', 'T4U', 'FTI', 'T3']

# Special handling for TSH (right-skewed distribution)
tsh_feature = ['TSH']


# Binary categorical features
binary_features = ['on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds',
                    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
                    'query_hypothyroid', 'query_hyperthyroid', 'lithium',
                    'goitre', 'tumor', 'hypopituitary', 'psych']

# categorical categorical features
categorical_features = ['sex', 'referral_source']

# Create specialized pipelines
age_pipeline = Pipeline([
    ('outlier_handler', AgeOutlierImputer(threshold=150)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

standard_numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

tsh_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log_transform', TSHLogTransformer()),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])


def get_preprocessor():
    return ColumnTransformer(
    transformers=[
        ('age', age_pipeline, ['age']),
        ('standard_numeric', standard_numeric_pipeline, standard_numeric),
        ('tsh', tsh_pipeline, tsh_feature),
        ('binary', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False), 
            binary_features),
        ('categorical', categorical_pipeline, categorical_features),
    ],
    remainder='drop'
)

def get_label_encoder(y_train, y_test):
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test, le

