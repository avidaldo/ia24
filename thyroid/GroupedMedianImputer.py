import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class GroupedMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, columns_to_impute):
        self.group_col = group_col
        self.columns_to_impute = columns_to_impute

    def fit(self, X, y=None):
        # Asegurarse de que la columna de agrupación existe
        if self.group_col not in X.columns:
            raise KeyError(f"'{self.group_col}' not found in input DataFrame")
        # Calcular la mediana para cada columna, agrupado por el grupo (sexo)
        self.medians_ = X.groupby(self.group_col)[self.columns_to_impute].median()
        return self

    def transform(self, X):
        X = X.copy()
        # Para cada grupo, rellenar los nulos en cada columna con la mediana correspondiente
        for group in X[self.group_col].unique():
            for col in self.columns_to_impute:
                if group in self.medians_.index:
                    X.loc[X[self.group_col] == group, col] = X.loc[X[self.group_col] == group, col].fillna(
                        self.medians_.loc[group, col]
                    )
        return X


# Example usage of GroupedMedianImputer
if __name__ == "__main__":
    data = {
        'Status': ['Developed', 'Developing', 'Developed', 'Developing', 'Developed', 'Developing'],
        'GDP': [50000, 15000, 52000, np.nan, 48000, 16000],
        'Inflation': [2.5, 5.0, np.nan, 4.5, 2.8, np.nan]
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)

    imputer = GroupedMedianImputer(group_col='Status', columns_to_impute=['GDP', 'Inflation'])
    df_imputed = imputer.fit_transform(df)
    print("\nDataFrame after GroupedMedianImputer:")
    print(df_imputed)

