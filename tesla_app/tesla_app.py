from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# ML imports
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

##############################################################################
# 1) Load CSV "database"
##############################################################################
DATA_PATH = "all_data_tesla.csv"
df = pd.read_csv(DATA_PATH)

# Ensure columns exist
if "Total Rainfall (mm)" not in df.columns:
    df["Total Rainfall (mm)"] = 0.0
if "Mean Temperature (°C)" not in df.columns:
    df["Mean Temperature (°C)"] = 0.0
if "Total Sunny Days" not in df.columns:
    df["Total Sunny Days"] = 0

##############################################################################
# 2) Weather fetching
##############################################################################
def get_weather_data_open_meteo(year: int, month: int, latitude=40.6401, longitude=22.9444):
    """
    Fetch daily data for the given (year, month) and aggregate.
    For simplicity, use 28 days for Feb, 30 for others.
    """
    days_in_month = 28 if month == 2 else 30

    total_temp = 0
    total_rainfall = 0
    total_sunny_days = 0
    valid_days = 0

    for day in range(1, days_in_month + 1):
        date = datetime(year, month, day)
        url = "https://archive-api.open-meteo.com/v1/era5"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": date.strftime('%Y-%m-%d'),
            "end_date": date.strftime('%Y-%m-%d'),
            "daily": "temperature_2m_mean,precipitation_sum,cloudcover_mean"
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data_json = resp.json()
                if "daily" in data_json:
                    daily_data = data_json["daily"]
                    avg_temp = daily_data.get("temperature_2m_mean", [None])[0]
                    precip_sum = daily_data.get("precipitation_sum", [0])[0]
                    cloud_mean = daily_data.get("cloudcover_mean", [100])[0]
                    sunny = (cloud_mean < 50)
                    if avg_temp is not None:
                        total_temp += avg_temp
                        total_rainfall += precip_sum
                        total_sunny_days += (1 if sunny else 0)
                        valid_days += 1
        except Exception as e:
            print(f"Error fetching data for {date}: {e}")

    mean_temp = total_temp / valid_days if valid_days > 0 else 0
    return {
        "Mean Temperature (°C)": mean_temp,
        "Total Rainfall (mm)": total_rainfall,
        "Total Sunny Days": total_sunny_days
    }

##############################################################################
# 3) Pydantic model for creating new records
##############################################################################
class NewRecord(BaseModel):
    SupplyNumber: str
    Year: int
    Month: int
    BillType: str
    TotalEnergyConsumed: Optional[float] = None
    MeanTemperature: float = 0.0
    TotalRainfall: float = 0.0
    TotalSunnyDays: int = 0

##############################################################################
# 4) FeatureEngineer and EnsembleEnergyForecaster
#    (Dropping "Bill Type" to avoid ValueError.)
##############################################################################
class FeatureEngineer:
    def __init__(self, max_lag=12):
        self.max_lag = max_lag
        self.month_lag_means_ = {}

    def get_params(self, deep=True):
        return {"max_lag": self.max_lag}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y=None):
        df_ = X.copy()
        df_ = self._create_features(df_)
        lag_cols = [c for c in df_.columns if 'lag_' in c or 'ewm_' in c or 'volatility_' in c]
        self.month_lag_means_ = (
            df_.groupby('Month')[lag_cols]
              .mean(numeric_only=True)
              .to_dict('index')
        )
        return self

    def transform(self, X, y=None):
        df_ = X.copy()
        df_ = self._create_features(df_)

        # Fill missing lag columns
        lag_cols = [c for c in df_.columns if 'lag_' in c or 'ewm_' in c or 'volatility_' in c]
        for idx, row in df_.iterrows():
            m = row['Month']
            for c in lag_cols:
                if pd.isna(row[c]) and (m in self.month_lag_means_):
                    if c in self.month_lag_means_[m]:
                        df_.at[idx, c] = self.month_lag_means_[m][c]

        # Fill remaining NAs with column means
        df_.fillna(df_.mean(numeric_only=True), inplace=True)

        # Drop columns we don't want the scaler to handle
        if 'Date' in df_.columns:
            df_.drop(columns=['Date'], inplace=True)

        # Also drop "Bill Type" if it's still there
        if 'Bill Type' in df_.columns:
            df_.drop(columns=['Bill Type'], inplace=True)

        # Also drop "Bill Type" if it's still there
        if 'Label' in df_.columns:
            df_.drop(columns=['Label'], inplace=True)

        return df_

    def _create_features(self, df_):
        df_['Date'] = pd.to_datetime(df_[['Year','Month']].assign(DAY=1))
        df_.sort_values(by=['Year','Month'], inplace=True)

        # Time-based features
        df_['quarter'] = df_['Date'].dt.quarter

        # Lags & EWM
        for i in [1, 2, 3, 6, 12]:
            if i <= self.max_lag:
                df_[f'lag_{i}'] = df_['Total KWh'].shift(i)
                df_[f'ewm_{i}'] = df_['Total KWh'].ewm(span=i).mean()

        # Rolling volatility
        for window in [3, 6, 12]:
            if window <= self.max_lag:
                rolling_std = df_['Total KWh'].rolling(window=window, min_periods=1).std()
                rolling_mean = df_['Total KWh'].rolling(window=window, min_periods=1).mean()
                df_[f'volatility_{window}'] = rolling_std / rolling_mean

        # Additional
        df_['temp_squared'] = df_['Mean Temperature (°C)'] ** 2
        df_['month_sin'] = np.sin(2 * np.pi * df_['Month'] / 12)
        df_['month_cos'] = np.cos(2 * np.pi * df_['Month'] / 12)

        # Ensure external columns exist
        if 'Total Rainfall (mm)' not in df_.columns:
            df_['Total Rainfall (mm)'] = 0.0
        if 'Total Sunny Days' not in df_.columns:
            df_['Total Sunny Days'] = 0

        return df_

class EnsembleEnergyForecaster:
    def __init__(self, do_grid_search=False, random_state=42):
        self.do_grid_search = do_grid_search
        self.random_state = random_state
        self.xgb_model_ = None
        self.rf_model_ = None
        self.scaler_ = StandardScaler()
        self.fitted_ = False
        self.alpha_xgb_ = 0.5
        self.alpha_rf_ = 0.5

    def get_params(self, deep=True):
        return {"do_grid_search": self.do_grid_search, "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        X_scaled = self.scaler_.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=3)

        if self.do_grid_search:
            # For demonstration
            param_dist_xgb = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
            }
            xgb_search = RandomizedSearchCV(
                XGBRegressor(random_state=self.random_state),
                param_distributions=param_dist_xgb,
                n_iter=5,
                cv=tscv,
                scoring='neg_mean_absolute_percentage_error',
                n_jobs=-1,
                random_state=self.random_state
            )
            xgb_search.fit(X_scaled, y)
            self.xgb_model_ = xgb_search.best_estimator_

            param_dist_rf = {
                'n_estimators': [100, 200],
                'max_depth': [6, 12],
                'max_features': ['sqrt', 0.75],
            }
            rf_search = RandomizedSearchCV(
                RandomForestRegressor(random_state=self.random_state),
                param_distributions=param_dist_rf,
                n_iter=5,
                cv=tscv,
                scoring='neg_mean_absolute_percentage_error',
                n_jobs=-1,
                random_state=self.random_state
            )
            rf_search.fit(X_scaled, y)
            self.rf_model_ = rf_search.best_estimator_
        else:
            self.xgb_model_ = XGBRegressor(n_estimators=200, learning_rate=0.05,
                                           max_depth=6, random_state=self.random_state)
            self.rf_model_  = RandomForestRegressor(n_estimators=200, max_depth=12,
                                                    random_state=self.random_state)
            self.xgb_model_.fit(X_scaled, y)
            self.rf_model_.fit(X_scaled, y)

        # Ensemble weighting
        val_size = int(0.1 * len(X))
        if val_size > 0:
            X_train_small, X_val = X_scaled[:-val_size], X_scaled[-val_size:]
            y_train_small, y_val = y[:-val_size], y[-val_size:]
            self.xgb_model_.fit(X_train_small, y_train_small)
            self.rf_model_.fit(X_train_small, y_train_small)

            pred_xgb = self.xgb_model_.predict(X_val)
            pred_rf  = self.rf_model_.predict(X_val)

            best_alpha = 0.5
            best_mape = float('inf')
            for alpha in np.linspace(0, 1, 21):
                combo = alpha * pred_xgb + (1 - alpha) * pred_rf
                cur_mape = mean_absolute_percentage_error(y_val, combo)
                if cur_mape < best_mape:
                    best_mape = cur_mape
                    best_alpha = alpha

            self.alpha_xgb_ = best_alpha
            self.alpha_rf_  = 1 - best_alpha

        # Refit on full data
        self.xgb_model_.fit(X_scaled, y)
        self.rf_model_.fit(X_scaled, y)
        self.fitted_ = True
        return self

    def predict(self, X, return_ci=True):
        if not self.fitted_:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        X_scaled = self.scaler_.transform(X)
        pred_xgb = self.xgb_model_.predict(X_scaled)
        pred_rf  = self.rf_model_.predict(X_scaled)
        predictions = self.alpha_xgb_ * pred_xgb + self.alpha_rf_ * pred_rf
        if return_ci:
            all_preds = np.vstack([pred_xgb, pred_rf])
            std_dev = np.std(all_preds, axis=0)
            ci_lower = predictions - 1.96 * std_dev
            ci_upper = predictions + 1.96 * std_dev
            return predictions, ci_lower, ci_upper
        return predictions

    # def plot_results(self, dates, y_true, y_pred, ci_lower=None, ci_upper=None):
    #     plt.figure(figsize=(10,5))
    #     plt.plot(dates, y_true, label='Actual', color='blue')
    #     plt.plot(dates, y_pred, label='Predicted', color='red')
    #     if ci_lower is not None and ci_upper is not None:
    #         plt.fill_between(dates, ci_lower, ci_upper, color='red', alpha=0.2, label='95% CI')
    #     plt.title('Energy Consumption Forecast')
    #     plt.xlabel('Date')
    #     plt.ylabel('Energy Consumption (kWh)')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.show()

##############################################################################
# 5) FastAPI, CORS
##############################################################################
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:5500"] etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

##############################################################################
# 6) Global pipeline
##############################################################################
pipeline = None
trained = False

def train_model_global():
    global pipeline, trained, df
    df_train = df.dropna(subset=["Total KWh"]).copy()
    y_train = df_train["Total KWh"]

    pipeline = Pipeline([
        ("features", FeatureEngineer(max_lag=12)),
        ("model", EnsembleEnergyForecaster(do_grid_search=False, random_state=42))
    ])
    pipeline.fit(df_train, y_train)
    trained = True
    print("Model training complete.")

##############################################################################
# 7) Routes (NO pagination)
##############################################################################

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/data")
def get_data_by_supply(supply_number: str):
    """
    Returns ALL records for the given supply_number, EXCEPT the last row.
    """
    # filtered = df[df["Supply Number"] == supply_number].copy()
    filtered = df[df["Supply Number"].astype(str) == str(supply_number)].copy()

    filtered.sort_values(by=["Year", "Month"], inplace=True)

    # Exclude last row if not empty
    if not filtered.empty:
        filtered = filtered.iloc[:-1]

    # Return the entire subset as a list
    return {
        "data": filtered.to_dict(orient="records")
    }

@app.get("/api/weather")
def get_weather(year: int, month: int):
    w = get_weather_data_open_meteo(year, month)
    return w

@app.post("/api/new_record")
def create_new_record(item: NewRecord):
    global df
    new_row = {
        "Supply Number": item.SupplyNumber,
        "Year": item.Year,
        "Month": item.Month,
        "Bill Type": item.BillType,
        "Total KWh": item.TotalEnergyConsumed if item.TotalEnergyConsumed else np.nan,
        "Mean Temperature (°C)": item.MeanTemperature,
        "Total Rainfall (mm)": item.TotalRainfall,
        "Total Sunny Days": item.TotalSunnyDays,
    }
    # df = df.append(new_row, ignore_index=True)
    # If you want to persist:
    # df.to_csv(DATA_PATH, index=False)
    return {"message": "New record added successfully."}

@app.get("/api/predict")
def predict_for_new_record(supply_number: str):
    global pipeline, trained, df
    # Train if not done
    if not trained or pipeline is None:
        train_model_global()

    subset = df[df["Supply Number"].astype(str) == str(supply_number)].copy()
    subset.sort_values(by=["Year","Month"], inplace=True)
    if subset.empty:
        return {"error": "No data found for given supply number."}

    # The last row is presumably the new record
    last_row = subset.iloc[[-1]].copy()

    # Ensure the target column exists (even if NaN)
    if "Total KWh" not in last_row.columns:
        last_row["Total KWh"] = np.nan

    X_for_pred = last_row.copy()
    preds, ci_lo, ci_hi = pipeline.predict(X_for_pred, return_ci=True)
    return {
        "Predicted kWh": float(preds[0]),
        "95pct_CI_Lower": float(ci_lo[0]),
        "95pct_CI_Upper": float(ci_hi[0])
    }
