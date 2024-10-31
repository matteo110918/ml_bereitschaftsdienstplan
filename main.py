# main.py

import pandas as pd
from src.feature_engineering import apply_feature_engineering  # Importiere die Feature-Engineering-Funktion


# Daten laden
def load_data(filepath):
    """Lädt die Rohdaten aus einer CSV-Datei."""
    return pd.read_csv(filepath, parse_dates=['date'])


def main():
    # Schritt 1: Daten laden
    data_filepath = 'data/raw/sickness_table.csv'  # Pfad zur Datei
    df = load_data(data_filepath)
    print("Daten geladen. Erste Zeilen:")
    print(df.head())

    # Schritt 2: Feature Engineering anwenden
    df_transformed = apply_feature_engineering(df, date_column='date')
    print("Feature Engineering abgeschlossen. Erste Zeilen des transformierten DataFrames:")
    print(df_transformed.head())


if __name__ == '__main__':
    main()
