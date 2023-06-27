import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

import os
import random
import numpy as np
import random as python_random


def read_data():
    return pd.read_csv('https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv')


def prepare_data(data):
    X=data.drop(["fetal_health"], axis=1)
    y=data["fetal_health"]

    columns_names = list(X.columns)
    scaler = preprocessing.StandardScaler()
    X_df = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_df, columns=columns_names)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)

    y_train = y_train -1
    y_test = y_test - 1
    return X_train, X_test, y_train, y_test


def create_model():
    grd_clf = GradientBoostingClassifier(max_depth=15,
                                         n_estimators=320,
                                         learning_rate=0.2)
    return grd_clf


def config_mlflow():
    MLFLOW_TRACKING_URI = ''
    MLFLOW_TRACKING_USERNAME = ''
    MLFLOW_TRACKING_PASSWORD = ''
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.sklearn.autolog(log_models=True,
                              log_input_examples=True,
                              log_model_signatures=True)


def train_model(model, X_train, y_train):
    with mlflow.start_run(run_name='experiment_01') as run:
      model.fit(X_train, y_train)


if __name__ == '__main__':
    config_mlflow()
    data = read_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    model = create_model()
    train_model(model, X_train, y_train)
