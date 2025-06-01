import pandas as pd
from sklearn.model_selection import train_test_split


def load_thyroid_3classes():
    file_path = "../data/thyroidDF.csv"
    df = pd.read_csv(file_path)

    # Simplify the target variable
    class_mapping = {
        '-': 'negative',
        'K': 'hyperthyroid', 'B': 'hyperthyroid', 'H|K': 'hyperthyroid',
        'KJ': 'hyperthyroid', 'GI': 'hyperthyroid',
        'G': 'hypothyroid', 'I': 'hypothyroid', 'F': 'hypothyroid', 'C|I': 'hypothyroid',
        'E': 'negative', 'LJ': 'negative', 'D|R': 'negative',
    }

    df['target'] = df['target'].map(class_mapping)

    df = df.dropna(subset=['target'])

    measured_cols = [col for col in df.columns if col.endswith('_measured')]

    columns_to_drop = [
        'patient_id',
        *measured_cols,  # Unpack the list of measured columns
        'TBG', # excessive missing values (96,6%)
    ]

    X = df.drop(columns_to_drop + ['target'], axis=1)
    y = df['target']


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test