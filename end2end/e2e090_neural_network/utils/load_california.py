import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def load_housing_data(filepath="../data/housing.csv",
                      test_size=0.2,
                      random_state=RANDOM_STATE) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    
    housing = pd.read_csv(filepath)
    
    housing["income_cat"] = pd.cut(housing["median_income"],
                                  bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                  labels=[1, 2, 3, 4, 5])
    
    strat_train_set, strat_test_set = train_test_split(
        housing, test_size=test_size, stratify=housing["income_cat"], random_state=random_state)
    
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    X_train = strat_train_set.drop("median_house_value", axis=1)
    y_train = strat_train_set["median_house_value"].copy()
    
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    
    
    return X_train, X_test, y_train, y_test 
