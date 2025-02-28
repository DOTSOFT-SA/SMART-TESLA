# SMART-TESLA
Open Code Repository for Project SMART TESLA

This a **FastAPI** application that manages energy consumption data from a CSV, fetches weather data, and provides machine learning predictions for future energy usage. This repository includes:

- A **FastAPI** backend (`tesla_app.py`) with:
  - CSV-based data loading
  - Weather data fetching (via Open Meteo)
  - ML pipeline (time-series feature engineering + ensemble regressor)
  - Several API endpoints (`/api/data`, `/api/weather`, `/api/new_record`, `/api/predict`)
- An **HTML/JS** frontend (`index.html`) that calls the API
- **Docker** support to containerize the app

## Features

- **Load data** from a CSV and store it in memory.
- **Display** all matching rows (except the last) by supply number in a simple table (no pagination).
- **Fetch weather data** automatically for a given month and year (aggregates daily data from Open Meteo).
- **Add new records** (e.g. for future months) with optional real or placeholder energy usage.
- **Train** or reuse an ML model pipeline (Random Forest + XGBoost ensemble) to **predict** next month’s energy consumption.
- **OpenAPI/Swagger** documentation automatically generated at `/docs`.

## Project Structure
├─ all_data.csv # Your CSV

├─ main.py # FastAPI app + ML classes + endpoints 

├─ templates/ 

│ └─ index.html # Frontend HTML/JS


## Requirements

- Python 3.9+ (or similar)
- Packages in `requirements.txt` (e.g., `fastapi`, `uvicorn`, `pandas`, `requests`, `scikit-learn`, `xgboost`, `jinja2`, etc.)
- (Optional) `Docker` if you want to build and run the container.

## Installation (Local)

1. **Clone** this repository:
   ```bash
   git clone https://github.com/username/energy-consumption-app.git
   cd energy-consumption-app
   ```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

License
MIT

Contributing
Feel free to submit PRs or open issues!
