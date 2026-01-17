from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from hydra import compose, initialize

from data_loader.creator import preprocess
from factory.trainer import Trainer
from models import MODELS
from path_definition import HYDRA_PATH

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class PredictionResult:
    symbol: str
    interval: str
    model: str
    predicted_mean: float
    last_close: float
    last_timestamp: str


@dataclass(frozen=True)
class DatasetInfo:
    symbol: str
    interval: str
    path: Path


def available_datasets(data_dir: Path = DATA_DIR) -> list[DatasetInfo]:
    datasets: list[DatasetInfo] = []
    for item in sorted(data_dir.glob("*-data.csv")):
        parts = item.stem.split("-")
        if len(parts) < 3:
            continue
        symbol = parts[0]
        interval = parts[1]
        datasets.append(DatasetInfo(symbol=symbol, interval=interval, path=item))
    return datasets


def resolve_dataset_path(symbol: str, interval: str, data_dir: Path = DATA_DIR) -> Path:
    normalized_symbol = symbol.upper()
    normalized_interval = interval.lower()
    candidate = data_dir / f"{normalized_symbol}-{normalized_interval}-data.csv"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find dataset for {normalized_symbol} at {normalized_interval}."
        )
    return candidate


def load_config(overrides: Optional[Iterable[str]] = None):
    with initialize(version_base=None, config_path=HYDRA_PATH):
        cfg = compose(config_name="train", overrides=list(overrides or []))
    return cfg


def load_price_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"timestamp": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def build_prediction(
    symbol: str,
    interval: str,
    model_type: str = "random_forest",
    data_dir: Path = DATA_DIR,
) -> PredictionResult:
    path = resolve_dataset_path(symbol, interval, data_dir)
    cfg = load_config([f"model={model_type}"])
    cfg.load_path = str(path)

    dataset_ = pd.read_csv(cfg.load_path)
    if "Date" not in dataset_.columns:
        dataset_ = dataset_.rename(columns={"timestamp": "Date"})
    if "High" not in dataset_.columns:
        dataset_ = dataset_.rename(columns={"high": "High"})
    if "Low" not in dataset_.columns:
        dataset_ = dataset_.rename(columns={"low": "Low"})

    dataset, _ = preprocess(dataset_, cfg)
    dataset = dataset.copy()
    dataset.drop(["predicted_high", "predicted_low"], axis=1, inplace=True)

    train_dataset = dataset[
        (dataset["Date"] > cfg.dataset_loader.train_start_date)
        & (dataset["Date"] < cfg.dataset_loader.train_end_date)
    ]
    if train_dataset.empty:
        train_dataset = dataset

    model = MODELS[cfg.model.type](cfg.model)
    Trainer(cfg, train_dataset, None, model).train()

    feature_frame = dataset.drop(["prediction"], axis=1)
    latest_features = feature_frame.tail(1)
    predicted_mean = float(model.predict(latest_features)[0])

    history = load_price_history(path)
    last_close = history["close"].iloc[-1] if "close" in history.columns else history["Close"].iloc[-1]
    last_timestamp = str(history["Date"].iloc[-1])

    return PredictionResult(
        symbol=symbol.upper(),
        interval=interval.lower(),
        model=model_type,
        predicted_mean=predicted_mean,
        last_close=float(last_close),
        last_timestamp=last_timestamp,
    )
