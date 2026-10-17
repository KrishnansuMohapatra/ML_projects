# House Price Prediction API

A FastAPI service that predicts house prices using a Random Forest model trained on the California Housing dataset.

## Features

- **JSON prediction** — `POST /predict` with a single house's features
- **Excel batch prediction** — `POST /predict/excel` uploads an `.xlsx` file and returns predictions in a downloadable spreadsheet

## Model

- Algorithm: `RandomForestRegressor` (100 estimators)
- Dataset: California Housing (`sklearn.datasets.fetch_california_housing`)
- Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python train.py   # trains and saves house_model.joblib + house_features.joblib
uv run python main.py    # starts the API on http://localhost:8000
```

## Usage

Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

**JSON prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.9841,
    "AveBedrms": 1.0238,
    "Population": 322.0,
    "AveOccup": 2.5556,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

**Excel batch prediction:**

```bash
curl -X POST http://localhost:8000/predict/excel \
  -F "file=@sample_input.xlsx" \
  -o predictions.xlsx
```

The uploaded file must contain columns: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude.
