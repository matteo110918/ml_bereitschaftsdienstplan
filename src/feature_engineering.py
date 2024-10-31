import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import holidays


# Custom Transformer für Datum-Features
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='date'):
        self.date_column = date_column

    def fit(self, X, y=None):
        # Dynamisch Feiertage für den Bereich des Datum-Features festlegen
        years = range(X[self.date_column].dt.year.min(), X[self.date_column].dt.year.max() + 1)
        self.de_holidays = holidays.Germany(years=years)
        return self

    def transform(self, X):
        X = X.copy()

        # Monat und andere Zeitfeatures erstellen
        X['month'] = X[self.date_column].dt.month
        X['day_of_week'] = X[self.date_column].dt.weekday.astype('category')  # 0=Monday, 6=Sunday
        X['quarter'] = X[self.date_column].dt.quarter.astype('category')

        # Berechnung der season, wobei month als int verwendet wird
        X['season'] = (X['month'].astype(int) % 12 // 3 + 1).astype('category')  # 1=Winter, 2=Spring, etc.

        # Konvertiere 'month' in Kategorie nach der Berechnung von season
        X['month'] = X['month'].astype('category')

        # Feiertag und Wochenende als boolsche Werte hinzufügen
        X['holiday'] = X[self.date_column].apply(lambda x: 1 if x in self.de_holidays else 0).astype(bool)
        X['weekend'] = X['day_of_week'].apply(lambda x: 1 if x >= 5 else 0).astype(bool)

        return X

# Custom Transformer für Lag- und Rolling-Features
class LagRollingFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, lag_features, rolling_features, lags=[7, 30], windows=[7, 30]):
        self.lag_features = lag_features
        self.rolling_features = rolling_features
        self.lags = lags
        self.windows = windows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for lag in self.lags:
            for feature in self.lag_features:
                X[f'{feature}_lag_{lag}'] = X[feature].shift(lag)

        for window in self.windows:
            for feature in self.rolling_features:
                X[f'{feature}_ma_{window}'] = X[feature].rolling(window=window).mean()
                X[f'{feature}_var_{window}'] = X[feature].rolling(window=window).var()

        # Entfernen von NaN-Werten, die durch Lag- und Rolling-Features entstehen
        X = X.dropna().reset_index(drop=True)
        return X


# Custom Transformer zum Entfernen der Datetime-Spalte
class DropDateColumn(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='date'):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[self.date_column])


# Funktion, um die Feature-Engineering-Pipeline anzuwenden
def apply_feature_engineering(df, date_column='date'):
    # Feature-Engineering-Pipeline
    feature_engineering_pipeline = Pipeline([
        ('date_features', DateFeatureExtractor(date_column=date_column)),
        ('lag_rolling_features', LagRollingFeatures(
            lag_features=['calls', 'n_sick', 'dafted'],
            rolling_features=['calls', 'n_sick', 'dafted']
        )),
        ('drop_date', DropDateColumn(date_column=date_column))
    ])

    # Anwendung der Pipeline auf den DataFrame
    df_transformed = feature_engineering_pipeline.fit_transform(df)
    return pd.DataFrame(df_transformed)