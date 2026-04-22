# Madrid Airbnb Host Intelligence
**Emerging Topics — University Project**

An end-to-end data science pipeline analysing the Madrid Airbnb market, with an interactive Streamlit dashboard for host price optimisation and guest sentiment analysis.

---

## Project Structure

```
Emerging Topics Project/
│
├── Final_Airbnb_Analysis_with_NLP.ipynb   ← Main notebook (run this)
├── Data/
│   ├── Airbnb Listings Data.csv
│   ├── Airbnb Reviews Data.csv
│   ├── Airbnb Calendar Data.csv
│   └── madrid-districts.geojson
│
├── dashboard/
│   ├── app.py                  ← Streamlit dashboard
│   ├── rf_regressor.pkl        ← Price prediction model
│   ├── rf_classifier.pkl       ← Demand classification model
│   ├── kmeans.pkl              ← K-Means clustering model
│   ├── cluster_scaler.pkl      ← Scaler for clustering
│   ├── occupancy_model.pkl     ← Occupancy prediction model
│   ├── occ_feature_cols.pkl    ← Feature columns for occupancy model
│   ├── meta.pkl                ← Feature metadata
│   ├── nn_scaler.pkl           ← Scaler for ANN model
│   ├── listings.csv            ← Cleaned listings export
│   ├── district_benchmark.csv  ← District-level price statistics
│   ├── listing_sentiment.csv   ← NLP sentiment & complaint results
│   ├── month_multipliers.csv   ← Seasonal demand multipliers
│   └── dow_multipliers.csv     ← Day-of-week demand multipliers
│
└── README.md
```

---

## Requirements

### Python Version
Python **3.12** (tested), 3.10+ should work.

### Install dependencies

Create and activate a virtual environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

Then install all required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn folium geopandas \
            vaderSentiment streamlit joblib shapely pyarrow
```

Key libraries and what they're used for:

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading and manipulation |
| `scikit-learn` | All ML models (Random Forest, KMeans, MLPRegressor, scalers) |
| `matplotlib`, `seaborn` | EDA and result visualisations |
| `geopandas`, `shapely` | Spatial join for district assignment |
| `folium` | Interactive geospatial map |
| `vaderSentiment` | Sentiment analysis on guest reviews |
| `streamlit` | Interactive dashboard |
| `joblib` | Saving/loading ML models |

---

## Data Setup

Place the following files in the `Data/` folder before running the notebook:

```
Data/
├── Airbnb Listings Data.csv
├── Airbnb Reviews Data.csv
├── Airbnb Calendar Data.csv
└── madrid-districts.geojson
```

> **Important:** The GeoJSON file must be loaded from the local path. Do not change the filename or load it from a URL — there is an SSL issue with the remote source.

Data source: [Inside Airbnb — Madrid](http://insideairbnb.com/get-the-data/)

---

## Running the Notebook

Open and run `Final_Airbnb_Analysis_with_NLP.ipynb` in Jupyter.

```bash
jupyter notebook "Final_Airbnb_Analysis_with_NLP.ipynb"
```

> **Important:** Cell 1 sets the working directory to the project root using `os.chdir()`. Run cells in order from top to bottom — later sections depend on variables defined earlier.

### Notebook sections in order:

1. **Setup** — working directory, imports
2. **Data Exploration** — load and inspect all three datasets
3. **Feature Engineering** — haversine distance, district spatial join
4. **NA Values** — imputation
5. **Outlier Analysis** — price and minimum nights capping
6. **EDA** — distributions, district medians, correlation heatmap
7. **Time Series** — SARIMA on monthly reviews + daily calendar occupancy
8. **Geospatial** — choropleth maps + Folium interactive map
9. **Clustering** — K-Means (k=4)
10. **Regression** — Linear, ANN, Random Forest price models
11. **Classification** — high-demand classifier
12. **NLP** — VADER sentiment + aspect-based complaint detection
13. **Occupancy Model** — Random Forest occupancy regression
14. **Dashboard Export** — saves all models and CSVs to `dashboard/`

---

## Running the Dashboard

The dashboard must be launched from the `dashboard/` folder. Run this from the project root:

```bash
cd dashboard && ../.venv/bin/streamlit run app.py --server.port 8501
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note:** Run the full notebook at least once before launching the dashboard — the `.pkl` model files and `.csv` exports must exist in `dashboard/` for the app to load.

### Demo listing IDs to try:
| ID | Name | What it shows |
|---|---|---|
| `5836616` | Justicia, the best Center | Dormant listing — underpriced and underbooked |
| `6332990` | Z45-Loft con Wifi en el Madrid Centro | Cleanliness complaints with review snippets |
| `229664` | — | All 7 complaint categories with high counts |

---

## Known Issues

- **GeoJSON SSL error:** Always load `madrid-districts.geojson` from the local `Data/` folder, never from a URL.
- **Streamlit cache:** If you re-run the notebook and regenerate the CSVs, restart the dashboard (`Ctrl+C` then re-run) to clear the cache.
- **Kernel order matters:** The NLP cell requires `reviews["compound"]` from the VADER cell above it. Always run cells in order.
