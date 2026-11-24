# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:49:53 2026

@author: Herakleitos
"""

import pandas as pd
import glob
import re
import requests
from time import sleep
import math
import numpy as np
import json
import jmespath
import geopandas as gpd


def create_frames_year_all():
    path = "./Data/IIEG/VENTA-RENTA/"
    files = glob.glob(path + "*.csv")
    sheets = ["Venta"]
    all_frames = []
    for i in range(2020, 2025):
        frames = []
        for file in files:
            if str(i) in file:
                print(file)

                match = re.search(r"_(\d+)_(\d{4}).xlsx$", file)

                if match:
                    month, year = match.groups()
                    month = month.lstrip("0")

                for s in sheets:
                    print(s)
                    raw = pd.read_excel(file, s)

                    if i in range(2020, 2022) or (
                        int(year) == 2022 and int(month) in range(0, 8)
                    ):
                        header_row = raw.index[raw.eq("Inmueble").any(axis=1)][0]
                        df = raw.iloc[header_row + 1 :].copy()
                        df.columns = raw.iloc[header_row]
                        df = df.reset_index(drop=True).dropna(how="all")

                    else:
                        df = raw

                    print(df.head())

                    df.rename(
                        columns={
                            "Inmueble": "Type",
                            "Precio": "Price",
                            "Municipio": "Municipality",
                            "Colonia": "Colony",
                            "Metros cuadrados de terreno": "Terrain",
                            "Metros cuadrados de construcción": "Construction",
                        },
                        inplace=True,
                    )

                    df.drop("Construction", axis=1, inplace=True, errors="Ignore")
                    df["Quarter"] = (int(month) + 2) // 3
                    df["Year"] = year
                    df["ppm2"] = df["Price"] / df["Terrain"]

                    text_cols = df.select_dtypes(include="string").columns

                    for column in text_cols:
                        df[column] = (
                            df[column]
                            .str.normalize("NFKD")
                            .str.encode("ascii", errors="ignore")
                            .str.decode("utf-8")
                            .str.strip()
                            .str.upper()
                        )

                    df.reset_index(inplace=True, drop=True)
                    df.dropna(inplace=True)

                    print(df.head())
                    frames.append(df)

        year = pd.concat(frames)

        all_frames.append(year)

    all_df = pd.concat(all_frames)

    all_df = all_df.loc[~(all_df.drop(columns=["Year", "Quarter"])).duplicated()]

    mask = (
        all_df["Terrain"].lt(20)
        & all_df["Terrain"].gt(1250)
        & all_df["ppm2"].gt(100000)
        & all_df["ppm2"].lt(2500)
    )

    all_df = all_df[~mask]
    all_df.dropna(inplace=True)
    all_df.to_csv("./Data/Clean/ALL_Venta.csv", index=False)


# def group_csv():
#     path = "./Data/Clean/Year_Quarter/"
#     files = glob.glob(path + "*.csv")
#     sheets = ["Venta"]

#     for s in sheets:
#         frames = list()
#         for file in files:
#             if s in file and "ALL" not in file:
#                 print(file)

#                 df = pd.read_csv(file)
#                 df.reset_index(inplace=True, drop=True)

#                 frames.append(df)

#         final = pd.concat(frames)
#         final.reset_index(inplace=True)
#         final.to_csv(f"./Data/Clean/ALL_{s}.csv", index=False)


def add_shf_index():
    path = "./Data/IIEG/VENTA-RENTA/"

    df = pd.read_csv("./Data/Clean/shf.csv")

    df = df.loc[df["Estado"] == "Jalisco"]

    df["Municipio"] = (
        df["Municipio"]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.strip()
        .str.upper()
    )

    df.rename(
        columns={
            "Año": "Year",
            "Municipio": "Municipality",
            "Trimestre": "Quarter",
            "Indice": "shf_index",
        },
        inplace=True,
    )

    df.loc[df["Municipality"] == "TLAJOMULCO DE ZUNIGA", "Municipality"] = "TLAJOMULCO"
    df.loc[df["Municipality"] == "SAN PEDRO TLAQUEPAQUE", "Municipality"] = "TLAQUEPAQUE"

    df.dropna(subset=["Municipality"], inplace=True)
    df.drop(columns="Global", inplace=True)
    df = df[df["Year"].gt(2019)]
    df.to_csv("./Data/Clean/shf.csv", index=False)

    # venta_df = pd.read_csv("./Data/Clean/ALL_Venta.csv")
    # venta_df.drop(columns=["shf_index"], inplace=True, errors="ignore")

    # venta_df = venta_df.merge(
    #     df[["Year", "Municipality", "Quarter", "shf_index"]],
    #     on=["Quarter", "Year", "Municipality"],
    #     how="left",
    # )

    # venta_df.dropna(inplace=True)
    # venta_df.to_csv("./Data/Clean/ALL_Venta.csv", index=False)


add_shf_index()


def make_cp_dict_unclean():
    path = "./Data/Clean/"
    files = glob.glob(path + "ALL_*.csv")

    frames = list()

    for file in files:
        print(file)
        df = pd.read_csv(file)

        uniques = df.drop_duplicates(subset=["Municipality", "Colony"])[
            ["Municipality", "Colony"]
        ]

        print(uniques.head())

        frames.append(uniques)

    cp_df = pd.concat(frames)
    cp_df.drop_duplicates(inplace=True)
    cp_df.to_csv("./Data/Process/helpers/cp_dict_unclean.csv", index=False)


def check_typos_cp_dict():

    import Levenshtein
    import pandas as pd

    cp = pd.read_csv("./Data/Process/helpers/cp_dict_unclean.csv")
    # Create an empty list to store the match scores
    scores = []

    for index1, row1 in cp.iterrows():
        for index2, row2 in cp.iterrows():
            # Calculate the match score using Levenshtein.distance
            if index1 > index2 or index1 == index2:
                continue

            score = Levenshtein.distance(row1["Colony"], row2["Colony"])
            if 0 < score < 4 and row1["Municipality"] == row2["Municipality"]:
                scores.append({"index1": index1, "index2": index2, "score": score})

    # Create a new DataFrame from the scores list
    scores_df = pd.DataFrame(scores)

    scores_df.to_csv("./Data/Process/helpers/cp_helper.csv")


def clean_sepomex_source():
    sepomex_df = pd.read_excel("./Data/CORREOS/Jalisco.xlsx", sheet_name=0)

    sepomex_df = sepomex_df.loc[
        sepomex_df["D_mnpio"].isin(
            [
                "Guadalajara",
                "Tonalá",
                "El Salto",
                "Zapopan",
                "San Pedro Tlaquepaque",
                "Tlajomulco de Zúñiga",
            ]
        ),
        ["d_codigo", "d_asenta", "D_mnpio"],
    ]

    sepomex_df.loc[sepomex_df["D_mnpio"] == "San Pedro Tlaquepaque", "D_mnpio"] = (
        "TLAQUEPAQUE"
    )
    sepomex_df.loc[sepomex_df["D_mnpio"] == "Tlajomulco de Zúñiga", "D_mnpio"] = (
        "TLAJOMULCO"
    )

    sepomex_df["D_mnpio"] = (
        sepomex_df["D_mnpio"]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.strip()
        .str.upper()
    )

    sepomex_df["d_asenta"] = (
        sepomex_df["d_asenta"]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.strip()
        .str.upper()
    )

    sepomex_df.rename(
        columns={"D_mnpio": "Municipality", "d_codigo": "CP", "d_asenta": "Colony"},
        inplace=True,
    )

    sepomex_df.to_csv("./Data/Process/helpers/sepomex.csv", index=False)


def clean_cp_dict():
    # Make replacement map with Claude, give cp_helper which is used for typos, give it
    # the sepomex.csv file and ISDCOL2020_2024.xlsx file which both contain the registered
    # colonies and their cp , ask it to merge numbered colonies without matches in sepomex
    # eg. TETLAN II -> TETLAN , in case TETLAN II is not in sepomex, and check typos
    # in edge cases where there is no match at all, use the cp given to match with the
    # nearest colony that appears in sepomex, use the web as a last resort. If there are
    # multiple matches within the same municipality, use the cp you want
    cp_dict = pd.read_csv("./Data/Process/helpers/cp_dict_unclean.csv")

    replace_df = pd.read_csv("./Data/Process/helpers/source_replacement_map.csv")
    replace_df.rename({"new_cp": "CP"}, inplace=True, axis=1)

    cp_dict = cp_dict.merge(
        replace_df[["Municipality", "original_colony", "new_colony", "CP"]],
        left_on=["Municipality", "Colony"],
        right_on=["Municipality", "original_colony"],
        how="left",
    )

    cp_dict["Colony"] = cp_dict["new_colony"]
    cp_dict.drop(["original_colony", "new_colony"], axis=1, inplace=True)
    cp_dict.drop_duplicates(subset=["Colony", "Municipality", "CP"], inplace=True)
    cp_dict.dropna(subset=["CP"], inplace=True)
    cp_dict["CP"] = cp_dict["CP"].astype("int32")
    cp_dict.to_csv("./Data/Clean/cp_dict.csv", index=False)


def clean_ALL_colonies():

    files = glob.glob("./Data/Clean/ALL_*.csv")
    replace_df = pd.read_csv("./Data/Process/helpers/source_replacement_map.csv")
    replace_df.rename({"new_cp": "CP"}, inplace=True, axis=1)

    for file in files:
        df = pd.read_csv(file)
        df = df.merge(
            replace_df[["Municipality", "original_colony", "CP"]],
            left_on=["Municipality", "Colony"],
            right_on=["Municipality", "original_colony"],
            how="left",
        )

        df.drop(["original_colony", "Colony"], axis=1, inplace=True)
        df.dropna(inplace=True)

        df = df.loc[~(df.drop(columns=["Year", "Quarter"])).duplicated()]
        df.to_csv(file, index=False)


# def mexpost():
#     cp_dict = pd.read_csv("./Data/Clean/cp_dict.csv")
#     cp_dict.dropna(subset=["CP"], inplace=True)
#     cp_dict["CP"] = cp_dict["CP"].astype("int32")

#     errors = list()
#     cp_match_only = list()

#     cp_dict["geometry"] = "PLACEHOLDER"
#     cp_dict["lat"] = 0.0
#     cp_dict["lon"] = 0.0

#     for index, row in cp_dict.iterrows():

#         cp = row["CP"]
#         colonia = row["Colony"]

#         response = requests.get(
#             f"http://localhost:8080/colonias/?cp={cp}&nombre={colonia}&incluir_geo=true"
#         )

#         if response.status_code == 200:
#             print(response.status_code, cp)

#             content = response.json()
#             if ("geometria" in content["resultados"][0]) == False:
#                 continue

#             if content["resultados"][0]["codigo"] != cp:
#                 print(
#                     f'CP NOT MATCHING!! SENT CP :{cp}, RECIEVED CP:{content["resultados"][0]["codigo"]}'
#                 )

#             cp_dict.loc[index, "geometry"] = content["resultados"][0]["geometria"]
#             cp_dict.loc[index, "lat"] = content["resultados"][0]["centro_lat"]
#             cp_dict.loc[index, "lon"] = content["resultados"][0]["centro_lon"]

#         elif response.status_code == 404:
#             response = requests.get(
#                 f"http://localhost:8080/colonias/?cp={cp}&incluir_geo=true"
#             )

#             if response.status_code == 200:
#                 print("NO NAME MATCH, PC ONLY: ", response.status_code, cp)

#                 if ("geometria" in content["resultados"][0]) == False:
#                     continue

#                 cp_dict.loc[index, "geometry"] = content["resultados"][0]["geometria"]
#                 cp_dict.loc[index, "lat"] = content["resultados"][0]["centro_lat"]
#                 cp_dict.loc[index, "lon"] = content["resultados"][0]["centro_lon"]

#             else:
#                 print("\t\t\t\t", response.status_code, cp)
#                 errors.append(cp)


def add_cp_coords_to_ALL():

    cp_dict = pd.read_csv("./Data/Clean/cp_dict.csv")
    file = "./Data/Clean/ALL_Venta.csv"

    df = pd.read_csv(file)
    df = df.merge(
        cp_dict[["Municipality", "CP", "lat", "lon"]],
        on=["Municipality", "CP"],
        how="left",
    )

    df.to_csv(file, index=False)


# def geo_json():
#     df = pd.read_csv("./Data/Clean/cp_dict_geo.csv")
#     features = list()

#     for index, row in df.iterrows():
#         if row["lat"] == 0:
#             continue

#         features.append(
#             {
#                 "type": "Feature",
#                 "properties": {
#                     "Municipality": row["Municipality"],
#                     "Colony": row["Colony"],
#                     "CP": row["CP"],  # or str(row["CP"]) to keep leading zeros
#                     "lat": row["lat"],
#                     "lon": row["lon"],
#                 },
#                 "geometry": json.loads(row["geometry"]),
#             }

#     feature_coll = {"type": "FeatureCollection", "features": features}

#     with open("colonies.geojson", "w", encoding="utf-8") as file:
#         json.dump(feature_coll, file)


def groupby_year():
    files = glob.glob("./Data/Clean/ALL*.csv")

    frames = []

    for file in files:
        df = pd.read_csv(file)

        for i in range(2020, 2025):
            rows = df.loc[df["Year"] == i]

            if file == "./Data/Clean\ALL_Venta.csv":
                rows.to_csv(f"./Data/Clean/VENTA_{i}.csv", index=False)
            else:
                rows.to_csv(f"./Data/Clean/RENTA_{i}.csv", index=False)


def robust_stats_terrain(frame, cols):
    g = frame.groupby(cols)["log_terrain"]
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median())
    n = g.transform("count")
    return med, mad, n


def robust_stats(frame, cols):
    g = frame.groupby(cols)["log_ppm2"]
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median())
    n = g.transform("count")
    return med, mad, n


def remove_price_terrain_outliers():
    file = "./Data/Clean/ALL_Venta.csv"
    df = pd.read_csv(file)
    # df.drop(columns="index", inplace=True)

    # Adding columns
    df["log_ppm2"] = np.log(df["ppm2"])
    df["robust_z"] = np.nan
    df["group_level"] = None

    # Necessary masks
    valid = df["Price"].gt(0) & df["Terrain"].gt(0)
    df["invalid_ratio"] = ~valid
    pending = valid.copy()

    Z_THRESHOLD = 3.5  # common cutoff for robust (MAD) z-scores
    MIN_GROUP = 15  # smallest group allowed to have its own statistics
    MIN_MAD = (
        0.10  # minimum spread (log units, ~10%) so near-identical groups don't explode
    )
    GROUP_LEVELS = [  # most specific level first, broadest last
        ["Type", "CP", "Year"],  # same postal code, same year
        ["Type", "CP"],  # same postal code, any year
    ]

    for cols in GROUP_LEVELS:
        med, mad, n = robust_stats(df, cols)
        ok = pending[valid] & (n >= MIN_GROUP)
        idx = ok[ok].index
        # Minimum spread: when most listings in a group share one price, MAD is
        # ~0 and tiny differences would look extreme. MIN_MAD = 0.10 in log units
        # means about ±10%.
        mad = mad.clip(lower=MIN_MAD)
        # 0.6745 rescales MAD so the score is comparable to a normal z-score
        df.loc[idx, "robust_z"] = 0.6745 * (df.loc[idx, "log_ppm2"] - med[idx]) / mad[idx]
        df.loc[idx, "group_level"] = "+".join(cols)
        pending.loc[idx] = False

    # --- 5. Flag ------------------------------------------------------------
    df["outlier"] = df["robust_z"].abs() > Z_THRESHOLD

    df = df[~df["outlier"]]
    df.drop(
        columns=["invalid_ratio", "group_level", "robust_z", "outlier", "log_ppm2"],
        inplace=True,
    )
    df.to_csv(file, index=False)


def remove_terrain_outliers():
    file = "./Data/Clean/ALL_Venta.csv"
    df = pd.read_csv(file)
    # df.drop(columns="index", inplace=True)

    # Adding columns
    df["log_terrain"] = np.log(df["Terrain"])
    df["robust_z"] = np.nan
    df["group_level"] = None

    # Necessary masks
    valid = df["Terrain"].gt(0)
    df["invalid_ratio"] = ~valid
    pending = valid.copy()

    Z_THRESHOLD = 3.5  # common cutoff for robust (MAD) z-scores
    MIN_GROUP = 15  # smallest group allowed to have its own statistics
    MIN_MAD = (
        0.10  # minimum spread (log units, ~10%) so near-identical groups don't explode
    )
    GROUP_LEVELS = [  # most specific level first, broadest last
        ["CP"]  # same postal code, any year
    ]

    for cols in GROUP_LEVELS:
        med, mad, n = robust_stats_terrain(df, cols)
        ok = pending[valid] & (n >= MIN_GROUP)
        idx = ok[ok].index
        # Minimum spread: when most listings in a group share one price, MAD is
        # ~0 and tiny differences would look extreme. MIN_MAD = 0.10 in log units
        # means about ±10%.
        mad = mad.clip(lower=MIN_MAD)
        # 0.6745 rescales MAD so the score is comparable to a normal z-score
        df.loc[idx, "robust_z"] = (
            0.6745 * (df.loc[idx, "log_terrain"] - med[idx]) / mad[idx]
        )
        df.loc[idx, "group_level"] = "+".join(cols)
        pending.loc[idx] = False

    # --- 5. Flag ------------------------------------------------------------
    df["outlier"] = df["robust_z"].abs() > Z_THRESHOLD

    df = df[~df["outlier"]]
    df.drop(
        columns=["invalid_ratio", "group_level", "robust_z", "outlier", "log_terrain"],
        inplace=True,
    )
    df.to_csv(file, index=False)


def add_geometry_from_14JAL():
    cp_dict = {}
    val2 = []

    with open("./Data/Clean/14-Jal.geojson", "r") as f:
        data = json.load(f)

        keys = jmespath.search("features[].properties.d_codigo", data)
        values = jmespath.search(str("features[].geometry"), data)
        for val in values:
            val2.append(str(val))

        cp_dict = dict(zip(keys, val2))
        df = pd.DataFrame.from_dict(cp_dict, orient="index", columns=["geometry"])
        df["CP"] = df.index
        df.reset_index(drop=True, inplace=True)

    cp_dict = pd.read_csv("./Data/Clean/cp_dict.csv")

    cp_dict["CP"] = cp_dict["CP"].astype("int32")
    # cp_dict.drop(
    #     columns=["lat", "lon", "geometry", "closest_lat", "closest_lon", "closest_gamma"],
    #     inplace=True,
    # )

    cp_dict = cp_dict.merge(df[["CP", "geometry"]], on=["CP"], how="left")
    cp_dict.drop_duplicates(subset=["CP"], inplace=True)
    cp_dict.drop(cp_dict.index[cp_dict["CP"] == 44739], inplace=True)

    cp = gpd.read_file("./Data/Clean/14-Jal.geojson")  # one shape per CP

    # Compute in meters (UTM zone 13N covers Guadalajara), then convert back to lat/lon
    metric = cp.to_crs("EPSG:32613")

    cent = metric.centroid.to_crs("EPSG:4326")  # geometric center
    inside = metric.representative_point().to_crs("EPSG:4326")  # always inside the shape

    # Use the centroid unless it falls outside the shape; then use the inside point
    outside = ~metric.centroid.within(metric.geometry)
    pt = cent.where(~outside, inside)

    cp["lat"] = pt.y
    cp["lon"] = pt.x

    cp_df = cp[["d_codigo", "lat", "lon"]]
    cp_df.rename({"d_codigo": "CP"}, inplace=True, axis=1)

    cp_dict = pd.read_csv("./Data/Clean/cp_dict.csv")
    cp_dict = cp_dict.merge(cp_df[["CP", "lat", "lon"]], on=["CP"], how="left")
    cp_dict.to_csv("./Data/Clean/cp_dict.csv", index=False)


def group_lonely_cps():
    cp_dict = pd.read_csv("./Data/Clean/cp_dict.csv")
    df = pd.read_csv("./Data/Clean/ALL_Venta.csv")
    cp_col = df["CP"]
    count = cp_col.value_counts(ascending=True)
    count = count.where(count == 1).dropna()

    replacement_map = dict()

    for item, index in count.items():

        for shift in range(1, 20):
            new_cp = item + shift

            if new_cp not in cp_dict["CP"].values or new_cp in count.values:

                new_cp = item - shift

                if new_cp not in cp_dict["CP"].values or new_cp in count.values:
                    continue

            replacement_map.update({item: new_cp})
            break

    cp_col = cp_col.replace(replacement_map)

    df["CP"] = cp_col
