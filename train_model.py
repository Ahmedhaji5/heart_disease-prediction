"""Reproduce the final model from the supplied Colab notebook.

Usage: python train_model.py --csv heart_disease_uci.csv
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
CATEGORICAL = ['sex', 'cp', 'fbs', 'restecg', 'exang']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='heart_disease_uci.csv')
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    required = NUMERIC + CATEGORICAL + ['num']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    df['chol'] = df['chol'].replace(0, np.nan)
    df['trestbps'] = df['trestbps'].replace(0, np.nan)
    y = (df['num'] > 0).astype(int)
    X = df[NUMERIC + CATEGORICAL]
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), NUMERIC),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL),
    ])
    model = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(C=0.1, max_iter=2000, random_state=42))])
    model.fit(X, y)
    output = Path(__file__).with_name('heart_disease_logistic_regression.pkl')
    joblib.dump(model, output)
    print(f'Saved {output} | training rows: {len(X)} | features: {X.shape[1]}')

if __name__ == '__main__':
    main()
