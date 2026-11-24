# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 09:54:28 2026

@author: Herakleitos
"""

from os import path
import pandas as pd
import tarfile
import requests
import matplotlib.pyplot as plt
import sys
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder as OHE
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
import glob


from dateutil.relativedelta import relativedelta

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

import joblib
import json

from math import radians, cos, sin, asin, sqrt


def dist(lat1, long1, lat2, long2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])
    # haversine formula
    dlon = long2 - long1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km


# N_CLUSTERS = 30  # try 20–60
# GAMMA_CL = 7500  # try 500–5000
SMOOTH = 10  # target-encoding smoothing (listings)
stats_df = pd.DataFrame(
    columns=["N_CLUSTERS", "GAMMA", "RIDGE_PARAMS", "R2_TEST", "RMSE_TEST"]
)

cp_dict = pd.read_csv("./Data/Clean/cp_dict.csv")

std_scaler_x = StandardScaler()
std_scaler_y = StandardScaler()


df = pd.read_csv("./Data/Clean/ALL_Venta.csv")

df = df.merge(cp_dict[["closest_gamma", "CP"]], on="CP", how="left")
df.drop(columns=["CP", "ppm2"], inplace=True)
df.dropna(inplace=True)

df["Terrain"] = np.log(df["Terrain"])
df["Price"] = np.log(df["Price"])

df = pd.get_dummies(df, columns=["Type", "Municipality"], dtype=float)

x_df = df.drop("Price", axis=1)
price_col = df[["Price"]]


N_CLUSTERS = 280
GAMMA_CL = 10000
km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
km.fit(x_df[["lat", "lon"]])
sim_cols = [f"cl_sim_{i}" for i in range(N_CLUSTERS)]
x_df[sim_cols] = rbf_kernel(x_df[["lat", "lon"]], km.cluster_centers_, gamma=GAMMA_CL)
x_df[sim_cols] = rbf_kernel(x_df[["lat", "lon"]], km.cluster_centers_, gamma=GAMMA_CL)

x_df = x_df.copy()
x_df["Price"] = price_col

x_train, x_test = train_test_split(x_df, test_size=0.20, random_state=42)

# x_train.to_csv("./Data/Clean/train.csv", index=False)
# x_test.to_csv("./Data/Clean/test.csv", index=False)

y_train = x_train[["Price"]]
x_train.drop(columns="Price", inplace=True)

y_test = x_test[["Price"]]
x_test.drop(columns="Price", inplace=True)

x_train = std_scaler_x.fit_transform(x_train)
x_test = std_scaler_x.transform(x_test)

y_train = std_scaler_y.fit_transform(y_train).ravel()
y_test = std_scaler_y.transform(y_test).ravel()

alphas = [10]

folds = KFold(n_splits=10, shuffle=True, random_state=42)

ridge_grid = GridSearchCV(
    estimator=Ridge(),
    param_grid={"alpha": alphas},
    scoring="r2",
    cv=folds,
    n_jobs=-1,
)

ridge_grid.fit(x_train, y_train)

modelo_final = ridge_grid.best_estimator_

print("N_CLUSTERS:", N_CLUSTERS, "\tGAMMA:", GAMMA_CL)
print("Mejores hiperparámetros:", ridge_grid.best_params_)
print("R2 CV train:", ridge_grid.best_score_)
print("Modelo final:", modelo_final)

y_pred_train = modelo_final.predict(x_train)
y_pred_test = modelo_final.predict(x_test)

ytrain_real = std_scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
ytest_real = std_scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

y_pred_train_real = std_scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
y_pred_test_real = std_scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()

metrics = {
    "N_CLUSTERS": N_CLUSTERS,
    "GAMMA": GAMMA_CL,
    "RIDGE_PARAMS": ridge_grid.best_params_,
    "R2_TEST": r2_score(ytest_real, y_pred_test_real),
    "RMSE_TEST": root_mean_squared_error(ytest_real, y_pred_test_real),
}

stats_df.loc[len(stats_df)] = metrics


def find_nearest(stations_df, lat, long):

    distances = stations_df.apply(
        lambda row: dist(lat, long, row["lat"], row["lon"]), axis=1
    )
    return stations_df.loc[distances.idxmin(), ["lat", "lon"]]


def get_nearest_station():
    stations_df = pd.read_csv("./Data/Clean/train_stations.csv")
    stations_df.drop(
        stations_df.index[stations_df["line"] == "Mi Macro Calzada"], inplace=True
    )

    stations_df.drop_duplicates("station", inplace=True)

    cp_df = pd.read_csv("./Data/Clean/cp_dict.csv")

    cp_df.drop(
        columns=["closest_lat", "closest_lon"],
        errors="ignore",
        inplace=True,
    )

    closest_cords = cp_df.apply(
        lambda row: find_nearest(stations_df, row["lat"], row["lon"]),
        axis=1,
        result_type="expand",
    )

    closest_cords.rename(
        columns={"lon": "closest_lon", "lat": "closest_lat"}, inplace=True
    )

    cp_df = pd.concat([cp_df, closest_cords], axis="columns")

    cp_df.to_csv("./Data/Clean/cp_dict.csv", index=False)


def kfb_kernel_trainlines():
    cp_df = pd.read_csv("./Data/Clean/cp_dict.csv")

    cp_df.drop(
        columns=["closest_gamma"],
        errors="ignore",
        inplace=True,
    )

    coords_df = (
        cp_df[["lat", "lon", "closest_lat", "closest_lon"]]
        .dropna(subset=["lat"])
        .drop_duplicates(subset=["lat", "lon"])
        .copy()
    )

    gamma = 5700
    d2 = (coords_df["lat"] - coords_df["closest_lat"]) ** 2 + (
        coords_df["lon"] - coords_df["closest_lon"]
    ) ** 2

    coords_df["closest_gamma"] = np.exp(-gamma * d2)
    cp_df = cp_df.merge(
        coords_df, on=["lat", "lon", "closest_lat", "closest_lon"], how="left"
    )
    cp_df.to_csv("./Data/Clean/cp_dict.csv", index=False)
