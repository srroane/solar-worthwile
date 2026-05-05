"""
Machine-learning and economic logic for solar production forecasting.

This module is intentionally UI-free so it can be imported by Streamlit,
notebooks, tests, or scheduled jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["ghi", "temperature_c", "cloud_cover_pct"]
TARGET_COLUMN = "solar_output_kwh"


@dataclass(frozen=True)
class PVSystemConfig:
    """Technical assumptions for the synthetic solar plant."""

    capacity_kwp: float = 8.0
    performance_ratio: float = 0.82
    temperature_coefficient: float = -0.004
    nominal_cell_temperature_c: float = 25.0
    inverter_limit_kw: float | None = 8.0


def _require_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def simulate_physical_output(
    weather_data: pd.DataFrame,
    config: PVSystemConfig | None = None,
) -> pd.Series:
    """
    Estimate hourly PV production with a compact physical approximation.

    Required columns:
    - ghi: Global Horizontal Irradiance in W/m2
    - temperature_c: Ambient temperature in degrees Celsius
    - cloud_cover_pct: Cloud cover from 0 to 100

    Returns hourly energy in kWh. For hourly rows, average kW over the hour is
    numerically equivalent to kWh.
    """

    _require_columns(weather_data, FEATURE_COLUMNS)
    cfg = config or PVSystemConfig()

    ghi = weather_data["ghi"].clip(lower=0).to_numpy(dtype=float)
    temperature_c = weather_data["temperature_c"].to_numpy(dtype=float)
    cloud_cover = weather_data["cloud_cover_pct"].clip(0, 100).to_numpy(dtype=float)

    irradiance_factor = ghi / 1000.0
    cloud_derate = 1.0 - 0.0045 * cloud_cover
    temperature_factor = 1.0 + cfg.temperature_coefficient * (
        temperature_c - cfg.nominal_cell_temperature_c
    )

    output_kw = (
        cfg.capacity_kwp
        * cfg.performance_ratio
        * irradiance_factor
        * cloud_derate
        * temperature_factor
    )
    output_kw = np.maximum(output_kw, 0.0)

    if cfg.inverter_limit_kw is not None:
        output_kw = np.minimum(output_kw, cfg.inverter_limit_kw)

    return pd.Series(output_kw, index=weather_data.index, name=TARGET_COLUMN)


def train_solar_model(
    training_data: pd.DataFrame,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline | RandomForestRegressor, dict[str, float]]:
    """
    Train a solar production model from historical or synthetic data.

    model_type can be "random_forest" or "linear_regression".
    Returns the fitted model and basic holdout metrics.
    """

    _require_columns(training_data, [*FEATURE_COLUMNS, TARGET_COLUMN])

    x = training_data[FEATURE_COLUMNS]
    y = training_data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    if model_type == "random_forest":
        model: Pipeline | RandomForestRegressor = RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "linear_regression":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
    else:
        raise ValueError("model_type must be 'random_forest' or 'linear_regression'")

    model.fit(x_train, y_train)
    predictions = np.maximum(model.predict(x_test), 0.0)

    metrics = {
        "mae_kwh": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }
    return model, metrics


def save_model(model: Pipeline | RandomForestRegressor, path: str | Path) -> None:
    """Persist a trained scikit-learn model for reuse by the Streamlit app."""

    joblib.dump(model, Path(path))


def load_model(path: str | Path) -> Pipeline | RandomForestRegressor:
    """Load a previously saved scikit-learn model."""

    return joblib.load(Path(path))


def predict_solar_output(
    weather_data: pd.DataFrame,
    model: Pipeline | RandomForestRegressor | None = None,
    model_path: str | Path | None = None,
    config: PVSystemConfig | None = None,
) -> pd.DataFrame:
    """
    Estimate hourly solar production in kWh.

    If a trained scikit-learn model or model_path is supplied, predictions use
    that model. Otherwise, the function falls back to the physical simulation.
    """

    _require_columns(weather_data, FEATURE_COLUMNS)
    result = weather_data.copy()

    if model is None and model_path is not None:
        model = load_model(model_path)

    if model is not None:
        predictions = np.maximum(model.predict(result[FEATURE_COLUMNS]), 0.0)
        result[TARGET_COLUMN] = predictions
        result["prediction_method"] = "machine_learning"
    else:
        result[TARGET_COLUMN] = simulate_physical_output(result, config=config)
        result["prediction_method"] = "physical_simulation"

    return result


def calculate_economics(
    energy_kwh: pd.Series | np.ndarray | list[float],
    price_per_kwh: float | dict[str, float],
    consumption_profile: pd.Series | np.ndarray | list[float],
) -> dict[str, float | pd.DataFrame]:
    """
    Calculate self-consumption, grid export, and total value of solar output.

    Economic formula:
    Revenue = (SelfConsumed * RetailRate) + (Exported * FeedInTariff)

    Self-consumed energy offsets electricity bought from the grid, so it is
    valued at the retail electricity rate. Exported surplus is sold to the grid,
    so it is valued at the feed-in tariff.
    """

    production = pd.Series(energy_kwh, dtype=float, name="production_kwh")
    consumption = pd.Series(
        consumption_profile, index=production.index, dtype=float, name="consumption_kwh"
    )

    if len(production) != len(consumption):
        raise ValueError("energy_kwh and consumption_profile must have the same length")

    if isinstance(price_per_kwh, dict):
        retail_rate = float(
            price_per_kwh.get("retail_rate", price_per_kwh.get("retail", 0.0))
        )
        feed_in_tariff = float(
            price_per_kwh.get("feed_in_tariff", price_per_kwh.get("export", retail_rate))
        )
    else:
        retail_rate = float(price_per_kwh)
        feed_in_tariff = float(price_per_kwh)

    # Self-consumption is production consumed immediately on site.
    self_consumed = np.minimum(production, consumption)
    # Surplus production is exported; unmet demand is imported from the grid.
    grid_export = np.maximum(production - consumption, 0.0)
    grid_import = np.maximum(consumption - production, 0.0)
    # Revenue = (SelfConsumed * RetailRate) + (Exported * FeedInTariff).
    savings = self_consumed * retail_rate
    export_revenue = grid_export * feed_in_tariff
    total_value = savings + export_revenue

    hourly = pd.DataFrame(
        {
            "production_kwh": production,
            "consumption_kwh": consumption,
            "self_consumed_kwh": self_consumed,
            "grid_export_kwh": grid_export,
            "grid_import_kwh": grid_import,
            "savings_value": savings,
            "export_revenue": export_revenue,
            "total_value": total_value,
        }
    )

    return {
        "hourly": hourly,
        "total_production_kwh": float(production.sum()),
        "total_consumption_kwh": float(consumption.sum()),
        "self_consumption_kwh": float(self_consumed.sum()),
        "grid_export_kwh": float(grid_export.sum()),
        "grid_import_kwh": float(grid_import.sum()),
        "self_consumption_rate": float(self_consumed.sum() / production.sum())
        if production.sum() > 0
        else 0.0,
        "self_sufficiency_rate": float(self_consumed.sum() / consumption.sum())
        if consumption.sum() > 0
        else 0.0,
        "bill_savings": float(savings.sum()),
        "feed_in_revenue": float(export_revenue.sum()),
        "total_revenue": float(total_value.sum()),
        "retail_rate": retail_rate,
        "feed_in_tariff": feed_in_tariff,
    }


def generate_synthetic_solar_data(
    output_csv: str | Path = "synthetic_solar_weather.csv",
    year: int = 2025,
    random_state: int = 42,
    config: PVSystemConfig | None = None,
) -> pd.DataFrame:
    """
    Generate one year of hourly weather and solar output data.

    The output is useful for development and for fitting a first-pass model
    before real site measurements are available.
    """

    rng = np.random.default_rng(random_state)
    index = pd.date_range(
        start=f"{year}-01-01 00:00:00",
        end=f"{year}-12-31 23:00:00",
        freq="h",
    )

    day_of_year = index.dayofyear.to_numpy()
    hour = index.hour.to_numpy()
    seasonal = np.sin(2 * np.pi * (day_of_year - 80) / 365.0).clip(0, 1)
    daylight = np.sin(np.pi * (hour - 6) / 12.0).clip(0, 1)

    clear_sky_ghi = 950 * daylight * (0.35 + 0.65 * seasonal)
    cloud_cover = rng.beta(2.0, 4.0, size=len(index)) * 100
    cloud_attenuation = 1.0 - 0.007 * cloud_cover
    ghi = np.maximum(clear_sky_ghi * cloud_attenuation + rng.normal(0, 35, len(index)), 0)

    annual_temperature = 10 + 11 * np.sin(2 * np.pi * (day_of_year - 172) / 365.0)
    daily_temperature = 4 * np.sin(2 * np.pi * (hour - 14) / 24.0)
    temperature_c = annual_temperature + daily_temperature + rng.normal(0, 2.5, len(index))

    weather = pd.DataFrame(
        {
            "timestamp": index,
            "ghi": ghi,
            "temperature_c": temperature_c,
            "cloud_cover_pct": cloud_cover,
        }
    ).set_index("timestamp")
    weather[TARGET_COLUMN] = simulate_physical_output(weather, config=config)
    weather[TARGET_COLUMN] = np.maximum(
        weather[TARGET_COLUMN] + rng.normal(0, 0.08, len(weather)), 0.0
    )

    output_path = Path(output_csv)
    weather.to_csv(output_path, index=True)
    return weather


def build_demo_model(
    csv_path: str | Path = "synthetic_solar_weather.csv",
    model_path: str | Path = "solar_rf_model.joblib",
    model_type: str = "random_forest",
) -> dict[str, float]:
    """
    Generate synthetic data if needed, train a model, save it, and return metrics.
    """

    csv_path = Path(csv_path)
    if csv_path.exists():
        data = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    else:
        data = generate_synthetic_solar_data(csv_path)

    model, metrics = train_solar_model(data, model_type=model_type)
    save_model(model, model_path)
    return metrics


if __name__ == "__main__":
    training_metrics = build_demo_model()
    print("Synthetic solar model trained.")
    print(training_metrics)
