import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans

RANDOM_STATE = 42


cat_pipeline = make_pipeline( # Pipeline para características categóricas
    SimpleImputer(strategy="most_frequent"),  # Imputa valores faltantes con el valor más frecuente
    OneHotEncoder(handle_unknown="ignore")    # Codifica características categóricas
)

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """
    Transformador que calcula la similitud entre cada instancia y los centroides de clusters
    utilizando un kernel RBF (Radial Basis Function).
    """
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters  # Número de clusters
        self.gamma = gamma            # Ancho de banda del kernel RBF
        self.random_state = random_state
        
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10, 
                             random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self
        
    def transform(self, X):
        # Calcula la similitud RBF entre cada instancia y los centroides de clusters
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


def column_ratio(X):
    """Calcula el ratio entre la primera y la segunda columna"""
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    """Función para nombrar las columnas de salida del ratio"""
    return ["ratio"]

def ratio_pipeline():
    """Pipeline que crea nuevas características dividiendo dos columnas"""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )

log_pipeline = make_pipeline( # Pipeline para transformación logarítmica
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)

default_num_pipeline = make_pipeline( # Pipeline por defecto para características numéricas
    SimpleImputer(strategy="median"),
    StandardScaler()
)

def get_preprocessing_pipeline(n_clusters=76, gamma=1.0):
    """
    Devuelve un pipeline de preprocesamiento configurado para los datos de viviendas

    Args:
        n_clusters: Número de clusters para la similitud geoespacial. Se usa por defecto 
            el valor que mejores resultados dio en la búsqueda de hiperparámetros.
        gamma: Parámetro del kernel RBF
        
    Returns:
        ColumnTransformer: Pipeline de preprocesamiento completo
    """
    cluster_simil = ClusterSimilarity(n_clusters=n_clusters, gamma=gamma, random_state=RANDOM_STATE)
    
    return ColumnTransformer([
        # Ratios (nuevas características)
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),        # ratio entre dormitorios y habitaciones
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),     # ratio entre habitaciones y hogares
        ("people_per_house", ratio_pipeline(), ["population", "households"]),     # ratio entre población y hogares
        
        # Transformación logarítmica para normalizar distribuciones sesgadas
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                              "households", "median_income"]),
        
        # Características geoespaciales utilizando clustering
        ("geo", cluster_simil, ["latitude", "longitude"]),
        
        # Características categóricas
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # Columnas restantes: housing_median_age


def scale_target(y_train, y_val, y_test):
    """
    Scale target variables using StandardScaler.
    
    Args:
        y_train: Training target values
        y_val: Validation target values  
        y_test: Test target values
        
    Returns:
        tuple: (scaled training data, scaled validation data, scaled test data, scaler)
    """
    y_scaler = StandardScaler()
    y_train_scaled_np = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_val_scaled_np = y_scaler.transform(y_val.values.reshape(-1, 1))
    y_test_scaled_np = y_scaler.transform(y_test.values.reshape(-1, 1))
    
    return y_train_scaled_np, y_val_scaled_np, y_test_scaled_np, y_scaler