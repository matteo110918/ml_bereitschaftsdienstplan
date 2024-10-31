# scaling_encoding.py

#### Hier wird dann nur mehr das ausgewählte Modell preprocessiert. #########

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_features_target(df, target_column='sby_need'):
    """Trennt die Zielvariable von den Features."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def get_preprocessor_lr_svr():
    """Erstellt und gibt die Pipeline für Lineare Regression und Support Vector Regression zurück."""
    return Pipeline(steps=[
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
                ('cat', OneHotEncoder(drop='first'), make_column_selector(dtype_include='category'))
            ], remainder='passthrough'))
    ])


def get_preprocessor_tree():
    """Erstellt und gibt die Pipeline für Tree-basierte Modelle zurück."""
    return Pipeline(steps=[
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('num', 'passthrough', make_column_selector(dtype_include=['int64', 'float64'])),
                ('cat', OneHotEncoder(drop='first'), make_column_selector(dtype_include='category'))
            ], remainder='passthrough'))
    ])


def apply_preprocessing(X, preprocessor):
    """Wendet die ausgewählte Pipeline auf die Features an."""
    return preprocessor.fit_transform(X)


def preprocess_data(df, target_column='sby_need'):
    """Hauptfunktion, die Feature-Skalierung und -Kodierung für verschiedene Modelltypen durchführt."""
    # Trenne die Zielvariable
    X, y = split_features_target(df, target_column=target_column)

    # Wähle die Pipelines
    preprocessor_lr_svr = get_preprocessor_lr_svr()
    preprocessor_tree = get_preprocessor_tree()

    # Wende Scaling und Encoding an
    X_scaled_encoded_lr_svr = apply_preprocessing(X, preprocessor_lr_svr)
    X_encoded_tree = apply_preprocessing(X, preprocessor_tree)

    return X_scaled_encoded_lr_svr, X_encoded_tree, y

