# Real Estate Price Estimator — Guadalajara Metropolitan Area (ZMG)

End-to-end MLOps project that predicts the asking price (MXN) of houses and apartments for sale in the Guadalajara Metropolitan Area. It covers data cleaning of public sources (IIEG, SHF, SEPOMEX/INE), geospatial feature engineering, a Ridge Regression model trained and tuned on **Amazon SageMaker**, and a serverless inference client (**API Gateway → Lambda → SageMaker endpoint**) served from an **S3 static website behind CloudFront**.

- **Live site:** https://d1a5bb1ypfqkfq.cloudfront.net/ (the CloudFront distribution and SageMaker endpoint are disabled when not in use to save costs, so predictions only work while the endpoint is in service)
- **Validation R²:** ~0.80

---

## Repository structure

| File | Description |
|---|---|
| `read.py` | Data cleaning pipeline: IIEG ingestion, SHF index join, colony/postal-code (CP) dictionary, geometry, and outlier removal. |
| `transform.py` | Feature engineering (nearest train station + RBF kernel), train/test split and local Ridge model. |
| `ISA2_venta_sagemaker.ipynb` | Notebook version of `read.py` + `transform.py` adapted to SageMaker: EDA, training script (`script.py`), inference script (`inference.py`), estimator, hyperparameter tuning and endpoint deployment. |
| `lambda_function.py` | AWS Lambda handler that builds the feature vector from a request and invokes the SageMaker endpoint. |
| `index.html` | Static website (prediction form + embedded Tableau Public map). |
| `map.ipynb` | Folium map of CP centroids coloured by the train-line RBF feature. |
| `Data.zip` | Raw and clean data (see below). |

### `Data.zip` contents

```
Data/
├── IIEG/VENTA-RENTA/        # Raw IIEG quarterly .xlsx files (BD_SECTOR_INMOBILIARIO_MM_YYYY.xlsx)
├── CORREOS/                 # SEPOMEX postal codes per municipality, INE colonies 2020-2024 (ISDCOL2020_2024.xlsx, shapefiles)
└── Clean/
    ├── ALL_Venta.csv        # Unified, cleaned sale listings
    ├── cp_dict.csv          # CP dictionary: colony, municipality, CP, lat/lon centroid, closest_gamma
    ├── shf.csv              # SHF housing price index (Jalisco)
    ├── 14-Jal.geojson       # Postal-code geometries for Jalisco
    ├── train_stations.csv   # Metropolitan rail stations
    ├── train.csv / test.csv # Model-ready splits
    ├── VENTA_YYYY.csv, Year/YYYY_Q_Venta.csv  # Per-year / per-quarter intermediate files
    └── mapa_bueno.twb       # Tableau workbook for the visualization tab
```

---

## Data sources

| Source | Content |
|---|---|
| **IIEG** (Instituto de Información Estadística y Geográfica de Jalisco) | Quarterly sale/rent listings 2019–2024 for Zapopan, Guadalajara, Tlaquepaque, Tlajomulco, Tonalá and El Salto (`.xlsx`). |
| **SHF** (Sociedad Hipotecaria Federal) | Quarterly SHF housing price index. For Jalisco it is published for Guadalajara, Zapopan, Tlajomulco and Tlaquepaque. |
| **SEPOMEX** and **INE Colonias 2024** | Colony ↔ postal code ↔ municipality catalogues. |
| **Jalisco postal-code GeoJSON** | Geometry of every postal-code area in Jalisco. |

---

## Pipeline

### 1. IIEG data wrangling (`create_frames_year_all`)

- 2019 data discarded (no terrain or colony info).
- `construction` column dropped (widespread inconsistencies).
- New column `ppm2 = Price / Terrain`.
- Rows with any NA dropped.
- Terrain kept in `[20, 1250]` m²; `ppm2` kept in `[2,500, 100,000]` MXN/m².
- All quarters merged into one file with `Year` and `Quarter` columns; listings repeated across quarters deduplicated.
- Strings normalized (accents removed) and upper-cased.

Output: `ALL_Venta.csv`

| Column | dtype | Description |
|---|---|---|
| `Type` | object | `CASA` \| `DEPARTAMENTO` |
| `Municipality` | object | Municipality of the property |
| `Colony` | object | Colony name |
| `Price` | float64 | Listed price (MXN) |
| `Terrain` | float64 | Listed terrain (m²) |
| `Quarter` | int64 | Quarter the listing was registered |
| `Year` | int64 | Year the listing was registered |
| `ppm2` | float64 | Price per m² |

### 2. SHF index (`add_shf_index`)

The SHF index is filtered to the municipalities/years present in the IIEG data and left-joined on Municipality and Year. Unmatched rows (El Salto and Tonalá, which have no SHF index) are dropped.

### 3. Postal code, colony and geometry dictionary

IIEG only provides colony and municipality, so the postal code (CP) is reconstructed:

1. Extract all unique `(Municipality, Colony)` pairs (`make_cp_dict_unclean`).
2. Group colonies whose names differ by 1–4 characters using **Levenshtein** distance to catch typos (`check_typos_cp_dict`).
3. Normalize the SEPOMEX and INE Colonias 2024 catalogues (`clean_sepomex_source`).
4. Load the Jalisco postal-code GeoJSON.
5. Build a replacement map (with Claude) from the typo dictionary and the three databases:
   - merge typo variants of the same colony;
   - cross-reference against the catalogues, collapsing numbered colonies (`Colony I`, `Colony II`, …) to the catalogue name;
   - if no name match exists, look the colony up on the web and map it to the closest CP available.
6. Merge the replacement map into the CP dictionary and the listings on `(Municipality, Colony)`, replacing the colony with its CP (`clean_cp_dict`, `clean_ALL_colonies`).
7. Attach each CP's geometry from the GeoJSON by querying it with **jmespath** (`add_geometry_from_14JAL`).
8. Compute each CP centroid (`lat`, `lon`) with **geopandas**.
9. Drop duplicated geometries (several CPs map to one geometry).
10. Add `lat`/`lon` to `ALL_Venta.csv` (`add_cp_coords_to_ALL`).
11. Drop the colony name from `ALL_Venta.csv` (only needed for the joins; it would add noise to the model).

### 4. Outlier removal

Prices are strongly skewed, so the filter works on logarithms with a **robust z-score** based on the median and MAD:

```
z = 0.6745 · (x − median) / MAD
```

Listings with `|z| > 3.5` are removed. Statistics are computed among comparable listings, so a price is only judged against its own neighbourhood.

- **`remove_price_terrain_outliers()`** — scores `log(ppm2)` within groups of `Type + CP + Year`; if the group has fewer than 15 listings it falls back to `Type + CP`. Groups below 15 listings at both levels are not scored. A minimum MAD of `0.10` (≈ ±10 %) prevents near-identical groups from flagging small differences.
- **`remove_terrain_outliers()`** — same robust z-score on `log(Terrain)` within each CP.

| Constant | Value |
|---|---|
| `Z_THRESHOLD` | 3.5 |
| `MIN_GROUP` | 15 |
| `MIN_MAD` | 0.10 |

### 5. Feature engineering (`transform.py`)

**Distance to public transport**

- **`get_nearest_station()`** — loads `train_stations.csv` (metropolitan rail lines, excluding the Mi Macro Calzada BRT and repeated stations) and, for each CP centroid, finds the nearest station using the **haversine** distance.
- **`kfb_kernel_trainlines()`** — converts that distance into a similarity with an RBF kernel:

  ```
  closest_gamma = exp(−γ · d²),   γ = 5,700,  d in degrees
  ```

  The feature is 1 at a station, ~0.5 at ~1.2 km and ~0.1 at ~2.2 km (walking distance to a line). The median across colonies is 0.16. It is stored in the CP dictionary.

**Remaining preparation:** log of price and terrain, one-hot encoding of `Type` and `Municipality`, and standardization of features and target with `StandardScaler`.

### 6. Train / test split

- `closest_gamma` joined from the CP dictionary.
- `ppm2` dropped (it is a direct function of the target `Price`).
- `Price` and `Terrain` log-transformed.
- `Municipality` and `Type` one-hot encoded with `pd.get_dummies()`.
- CPs with a single listing are reassigned to the numerically nearest CP in the dictionary (`group_lonely_cps`) so the split can be stratified.
- Two splits — train/test and train/validation — both with `test_size=0.20`, `random_state=42`, `stratify=CP`.
- `CP` dropped afterwards (location is represented by `lat`/`lon` and the cluster features).

---

## Model (Amazon SageMaker)

Everything up to the split runs locally; training, tuning and deployment run on SageMaker (region `us-east-2`).

### Training (`script.py`)

- **Model:** Ridge Regression (scikit-learn, SageMaker SKLearn framework `1.4-2`, `ml.c5.2xlarge`).
- **Target:** `log(Price)`.
- **Location clusters:** KMeans on `lat`/`lon`, then RBF similarity of each listing to every cluster centre (`cl_sim_0 … cl_sim_{n-1}`), adding ~280 columns. This is done **inside the training script** so that:
  1. KMeans is fitted only on training rows (no data leakage);
  2. its hyperparameters can be tuned together with Ridge.
- `StandardScaler` on X and y.
- **Metrics:** R², RMSE (log scale) and MAE (MXN).
- Model, scalers and KMeans saved with `joblib.dump()`.

Hyperparameters exposed by the script: `--alpha`, `--n-clusters`, `--gamma-cl`.

### Hyperparameter tuning

SageMaker `HyperparameterTuner`, objective **maximize `validation-r2`** (parsed from the log line `Validation R2: …`).

| Hyperparameter | Range |
|---|---|
| `alpha` | 1 – 1000 (log scale) |
| `n-clusters` | 100 – 314 |
| `gamma-cl` | 1000 – 20000 (log scale) |

Results:

| alpha | gamma-cl | n-clusters | Validation R² |
|---|---|---|---|
| **1.3007** | **15810.6** | **301** | **0.8077** |
| 4.4338 | 6851.8 | 314 | 0.8076 |
| 624.0186 | 10538.4 | 258 | 0.8075 |
| 621.3798 | 7497.3 | 258 | 0.8074 |
| 4.4810 | 3944.7 | 155 | 0.8071 |

### Deployment

The best training job's artifacts (`model.tar.gz`) are loaded into an `SKLearnModel` with `inference.py` and deployed to a real-time endpoint:

- **Endpoint name:** `Price-SG-ENDPOINT`
- **Instance:** `ml.c5.2xlarge`
- **Deserializer:** `JSONDeserializer`
- **Validation R²:** ~0.80

---

## Inference client architecture

```
Client ─▶ CloudFront ─▶ S3 static website ─▶ API Gateway ─▶ Lambda ─▶ SageMaker endpoint
                                                              │
                                                              └─ reads cp_dict.csv, shf.csv from S3
```

### Lambda (`lambda_function.py`)

Accepts a `POST` request with a JSON body (a `GET` with the same fields as query parameters is also supported):

```json
{
  "request": {
    "Municipality": "ZAPOPAN",
    "CP": 45110,
    "Year": 2025,
    "Quarter": 2,
    "Terrain": 120,
    "Type": "CASA"
  }
}
```

The handler:

1. Loads `cp_dict.csv` and `shf.csv` from S3.
2. Takes `log(Terrain)`, one-hot encodes `Type` and `Municipality`.
3. Joins `lat`, `lon`, `closest_gamma` by CP and `shf_index` by `(Year, Quarter, Municipality)`.
4. Orders the columns exactly as in training:
   `Terrain, Quarter, Year, shf_index, lat, lon, closest_gamma, Type_Casa, Type_Departamento, Type_casa, Municipality_GUADALAJARA, Municipality_TLAJOMULCO, Municipality_TLAQUEPAQUE, Municipality_ZAPOPAN`
5. Invokes the endpoint and returns `{"prediction": ...}`.

No authentication is implemented. The endpoint name is read from the `ENDPOINT_NAME` environment variable.

### API Gateway

A single `POST` route with CORS enabled (the CloudFront site and the API are on different domains).

### S3 static website (`index.html`)

Single HTML page with two tabs:

- **Prediction** — dropdowns for Municipality, CP (must exist in the CP dictionary), Year (2020–2026, the years with an SHF index), Quarter and Type; free numeric input for Terrain. Returns the predicted price in MXN.
- **Visualization** — embedded Tableau Public map built from the postal-code GeoJSON, showing the 75th-percentile price per CP, with hover details (CP, colony) and price/municipality filters.

### CloudFront

CDN distribution with the S3 bucket as origin: https://d1a5bb1ypfqkfq.cloudfront.net/

---

## Running it

1. Unzip `Data.zip` in the repository root so paths resolve as `./Data/...`.
2. **Local cleaning and features:** run the functions in `read.py`, then `transform.py`, in this order:

   ```python
   create_frames_year_all(); add_shf_index()
   make_cp_dict_unclean(); check_typos_cp_dict(); clean_sepomex_source()
   clean_cp_dict(); clean_ALL_colonies()
   add_geometry_from_14JAL(); add_cp_coords_to_ALL()
   remove_price_terrain_outliers(); remove_terrain_outliers()
   get_nearest_station(); kfb_kernel_trainlines()
   ```

   `clean_cp_dict()` requires the replacement map `source_replacement_map.csv` in `./Data/Process/helpers/`.
3. **SageMaker:** open `ISA2_venta_sagemaker.ipynb` in a SageMaker notebook, set `bucket_name`, upload the clean data to S3, and run the training, tuning and deployment sections.
4. **Client:** create the Lambda from `lambda_function.py` (set `ENDPOINT_NAME`, give it S3 read and `sagemaker:InvokeEndpoint` permissions), expose it through API Gateway, set `API_URL` in `index.html`, and host the page on S3 + CloudFront.

### Main dependencies

`pandas`, `numpy`, `scikit-learn`, `geopandas`, `jmespath`, `python-Levenshtein`, `matplotlib`, `seaborn`, `folium`, `boto3`, `sagemaker`.

---

## Known limitations

- Data are asking prices scraped from public listings, likely biased towards more expensive properties that stay listed longer.
- El Salto and Tonalá are excluded because they have no SHF index.
