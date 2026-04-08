from __future__ import annotations

import gc
import math
import os
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .DataIO import read_tabular_frame
except ImportError:
    from DataIO import read_tabular_frame

try:
    from .ForecastStyle import ForecastFigureStyle
except ImportError:
    from ForecastStyle import ForecastFigureStyle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Photo" / "plots_nhits_darts"
OUTPUT_CSV = OUTPUT_DIR / "predictions_last_week_nhits.csv"
METRICS_CSV = OUTPUT_DIR / "metrics_by_horizon_nhits.csv"


class NHitsError(RuntimeError):
    pass


class DependencyError(NHitsError):
    pass


@dataclass(frozen=True, slots=True)
class NHitsSettings:
    input_chunk_length: int = 720
    output_chunk_length: int = 24
    test_horizon: int = 168
    calibration_horizon: int = 168
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_size: int = 128
    num_stacks: int = 3
    num_blocks: int = 3
    num_layers: int = 1
    dropout: float = 0.10
    random_state: int = 42
    early_stop: bool = True
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1e-4
    use_temperature: bool = True

    @property
    def validation_horizon(self) -> int:
        return max(self.input_chunk_length + self.output_chunk_length, 360)


@dataclass(frozen=True, slots=True)
class DataColumns:
    start: str
    end: str | None
    consumption: str
    temperature: str | None


@dataclass(frozen=True, slots=True)
class RuntimeConfiguration:
    accelerator: str
    devices: int | None
    precision: str
    num_workers: int

    @property
    def dataloader_kwargs(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "pin_memory": self.accelerator == "gpu" and self.num_workers > 0,
            "persistent_workers": False,
        }


@dataclass(frozen=True, slots=True)
class DartsDependencies:
    torch: Any
    TimeSeries: Any
    NHiTSModel: Any
    Scaler: Any
    mae: Any
    rmse: Any
    mape: Any
    r2_score: Any
    EarlyStopping: Any


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    series: Any
    temperature_series: Any
    scaler_y: Any
    scaler_temperature: Any
    model: Any
    dependencies: DartsDependencies


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    output_dir: Path
    forecast_csv_path: Path
    metrics_csv_path: Path
    figure_paths: tuple[Path, ...]


class PlotStyle:
    @staticmethod
    def apply() -> None:
        ForecastFigureStyle.apply()


class DependencyLoader:
    @staticmethod
    def load() -> DartsDependencies:
        try:
            import torch
            from darts import TimeSeries
            from darts.dataprocessing.transformers import Scaler
            from darts.metrics import mae, mape, r2_score, rmse
            from darts.models import NHiTSModel
            from pytorch_lightning.callbacks.early_stopping import EarlyStopping
        except ImportError as exc:
            raise DependencyError("nhits dependencies are unavailable") from exc

        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        return DartsDependencies(
            torch=torch,
            TimeSeries=TimeSeries,
            NHiTSModel=NHiTSModel,
            Scaler=Scaler,
            mae=mae,
            rmse=rmse,
            mape=mape,
            r2_score=r2_score,
            EarlyStopping=EarlyStopping,
        )


class RuntimeResolver:
    def __init__(self, dependencies: DartsDependencies) -> None:
        self.dependencies = dependencies

    def resolve(self) -> RuntimeConfiguration:
        torch = self.dependencies.torch
        use_gpu = bool(torch.cuda.is_available())

        if use_gpu:
            precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
            accelerator = "gpu"
            devices = 1
        else:
            precision = "32-true"
            accelerator = "cpu"
            devices = None

        is_windows = (os.name == "nt") or ("windows" in platform.system().lower())
        cpu_total = os.cpu_count() or 2
        num_workers = 0 if is_windows else max(1, min(4, cpu_total - 1))

        return RuntimeConfiguration(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            num_workers=num_workers,
        )


class DataLocator:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def find_source_data_path(self) -> Path:
        candidates = (
            self.project_root / "Data.xlsx",
            self.project_root / "Data.csv",
            self.project_root / "Data" / "Data.xlsx",
            self.project_root / "Data" / "Data.csv",
        )
        direct_match = self._first_existing_path(candidates)
        if direct_match is not None:
            return direct_match

        for pattern in ("Data.xlsx", "Data.csv"):
            matches = sorted(self.project_root.rglob(pattern))
            if matches:
                return matches[0]

        raise NHitsError("source data file not found")

    @staticmethod
    def _first_existing_path(candidates: tuple[Path, ...]) -> Path | None:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None


class SourceDataLoader:
    def __init__(self, locator: DataLocator) -> None:
        self.locator = locator

    def load(self) -> pd.DataFrame:
        source_path = self.locator.find_source_data_path()
        frame = self._read_frame(source_path)
        columns = self._resolve_columns(frame)

        result = frame.copy()
        result["start_dt"] = pd.to_datetime(result[columns.start], dayfirst=True, errors="coerce")

        if columns.end is not None:
            result["end_dt"] = pd.to_datetime(result[columns.end], dayfirst=True, errors="coerce")
        else:
            result["end_dt"] = result["start_dt"] + pd.Timedelta(hours=1)

        result["consumption_mwh"] = result[columns.consumption].map(self._coerce_numeric)

        if columns.temperature is not None:
            result["temperature_c"] = result[columns.temperature].map(self._coerce_numeric)
        else:
            result["temperature_c"] = np.nan

        result = result.loc[result["start_dt"].notna() & result["consumption_mwh"].notna()].copy()
        result["start_dt"] = result["start_dt"].dt.floor("h")

        if result.duplicated(subset="start_dt").any():
            result = (
                result.groupby("start_dt", as_index=False)
                .agg(
                    consumption_mwh=("consumption_mwh", "mean"),
                    temperature_c=("temperature_c", "mean"),
                    end_dt=("end_dt", "max"),
                )
                .sort_values("start_dt")
                .reset_index(drop=True)
            )
        else:
            result = (
                result.sort_values("start_dt")
                .drop_duplicates(subset="start_dt", keep="last")
                .reset_index(drop=True)
            )

        if result.empty:
            raise NHitsError("source dataset is empty after preprocessing")

        result = self._regularize_frame(result)
        if result.empty:
            raise NHitsError("source dataset is empty after regularization")

        return result[["start_dt", "end_dt", "consumption_mwh", "temperature_c"]]

    @staticmethod
    def _read_frame(source_path: Path) -> pd.DataFrame:
        return read_tabular_frame(source_path)

    def _resolve_columns(self, frame: pd.DataFrame) -> DataColumns:
        column_names = frame.columns.tolist()
        start_column = self._find_column(column_names, ("начало", "start")) or column_names[0]
        end_column = self._find_column(column_names, ("конец", "end"))
        consumption_column = self._find_column(
            column_names,
            ("потреб", "consum", "load"),
        )
        temperature_column = self._find_column(
            column_names,
            ("темпер", "temp"),
        )

        if consumption_column is None:
            raise NHitsError("consumption column not found")

        return DataColumns(
            start=start_column,
            end=end_column,
            consumption=consumption_column,
            temperature=temperature_column,
        )

    @staticmethod
    def _find_column(column_names: list[str], tokens: tuple[str, ...]) -> str | None:
        lowered_names = [name.lower() for name in column_names]
        for index, name in enumerate(lowered_names):
            if any(token in name for token in tokens):
                return column_names[index]
        return None

    @staticmethod
    def _coerce_numeric(value: Any) -> float:
        if value is None:
            return float("nan")
        if isinstance(value, (int, float, np.number)):
            if isinstance(value, float) and math.isnan(value):
                return float("nan")
            return float(value)

        text = str(value).strip().replace(" ", "").replace("\u00A0", "").replace(",", ".")
        if not text:
            return float("nan")

        try:
            return float(text)
        except ValueError:
            return float("nan")

    @staticmethod
    def _regularize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        regularized = (
            frame.set_index("start_dt")
            .reindex(pd.date_range(frame["start_dt"].min(), frame["start_dt"].max(), freq="h"))
            .rename_axis("start_dt")
            .reset_index()
        )
        regularized["end_dt"] = regularized["start_dt"] + pd.Timedelta(hours=1)
        regularized["consumption_mwh"] = regularized["consumption_mwh"].ffill()

        if regularized["temperature_c"].notna().any():
            regularized["temperature_c"] = regularized["temperature_c"].ffill()
            first_temperature_index = regularized["temperature_c"].first_valid_index()
            if first_temperature_index is not None:
                regularized = regularized.loc[first_temperature_index:].copy()

        first_target_index = regularized["consumption_mwh"].first_valid_index()
        if first_target_index is None:
            return regularized.iloc[0:0].copy()

        regularized = regularized.loc[first_target_index:].reset_index(drop=True)
        return regularized


class TimeSeriesBuilder:
    def __init__(self, dependencies: DartsDependencies, settings: NHitsSettings) -> None:
        self.dependencies = dependencies
        self.settings = settings

    def build(self, frame: pd.DataFrame) -> tuple[Any, Any]:
        TimeSeries = self.dependencies.TimeSeries

        series = TimeSeries.from_dataframe(
            frame,
            time_col="start_dt",
            value_cols="consumption_mwh",
            freq="H",
            fill_missing_dates=False,
        ).astype(np.float32)

        temperature_series = None
        if (
            self.settings.use_temperature
            and "temperature_c" in frame.columns
            and frame["temperature_c"].notna().any()
        ):
            temperature_series = TimeSeries.from_dataframe(
                frame[["start_dt", "temperature_c"]],
                time_col="start_dt",
                value_cols="temperature_c",
                freq="H",
                fill_missing_dates=False,
            ).astype(np.float32)

        return series, temperature_series


class NHitsTrainer:
    def __init__(
        self,
        dependencies: DartsDependencies,
        settings: NHitsSettings,
        runtime: RuntimeConfiguration,
    ) -> None:
        self.dependencies = dependencies
        self.settings = settings
        self.runtime = runtime

    def fit(self, series: Any, temperature_series: Any) -> TrainingArtifacts:
        holdout_horizon = (
            self.settings.validation_horizon
            + self.settings.calibration_horizon
            + self.settings.test_horizon
        )
        if len(series) <= holdout_horizon:
            raise NHitsError("insufficient history for train, validation, calibration and test splits")

        validation_start = len(series) - holdout_horizon
        validation_end = validation_start + self.settings.validation_horizon

        train_series = series[:validation_start]
        validation_series = series[validation_start:validation_end]

        if len(train_series) == 0 or len(validation_series) == 0:
            raise NHitsError("invalid training or validation split")

        Scaler = self.dependencies.Scaler
        scaler_y = Scaler()
        train_series_scaled = scaler_y.fit_transform(train_series)

        scaler_temperature = None
        train_temperature_scaled = None
        validation_temperature_scaled = None

        if temperature_series is not None:
            train_temperature = temperature_series[:validation_start]
            validation_temperature = temperature_series[validation_start:validation_end]
            scaler_temperature = Scaler()
            train_temperature_scaled = scaler_temperature.fit_transform(train_temperature)
            validation_temperature_scaled = scaler_temperature.transform(validation_temperature)

        callbacks = []
        if self.settings.early_stop:
            callbacks.append(
                self.dependencies.EarlyStopping(
                    monitor="val_loss",
                    patience=self.settings.early_stopping_patience,
                    mode="min",
                    min_delta=self.settings.early_stopping_delta,
                )
            )

        torch = self.dependencies.torch
        model = self.dependencies.NHiTSModel(
            input_chunk_length=self.settings.input_chunk_length,
            output_chunk_length=self.settings.output_chunk_length,
            num_stacks=self.settings.num_stacks,
            num_blocks=self.settings.num_blocks,
            num_layers=self.settings.num_layers,
            layer_widths=self.settings.hidden_size,
            dropout=self.settings.dropout,
            batch_size=self.settings.batch_size,
            n_epochs=self.settings.epochs,
            loss_fn=torch.nn.SmoothL1Loss(beta=1.0),
            optimizer_kwargs={
                "lr": self.settings.learning_rate,
                "weight_decay": self.settings.weight_decay,
            },
            random_state=self.settings.random_state,
            add_encoders={
                "datetime_attribute": {"past": ["hour", "dayofweek", "month"]},
                "cyclic": {"past": ["hour", "month"]},
                "position": {"past": ["relative"]},
                "transformer": self.dependencies.Scaler(),
                "tz": "Europe/Moscow",
            },
            pl_trainer_kwargs={
                "accelerator": self.runtime.accelerator,
                "devices": self.runtime.devices,
                "precision": self.runtime.precision,
                "callbacks": callbacks,
                "enable_model_summary": False,
                "log_every_n_steps": 50,
                "num_sanity_val_steps": 0,
                "deterministic": False,
                "gradient_clip_val": 0.5,
                "max_epochs": self.settings.epochs,
            },
            save_checkpoints=False,
            show_warnings=False,
            force_reset=True,
        )

        model.fit(
            series=train_series_scaled,
            past_covariates=train_temperature_scaled,
            val_series=scaler_y.transform(validation_series),
            val_past_covariates=validation_temperature_scaled,
            verbose=False,
            dataloader_kwargs=self.runtime.dataloader_kwargs,
        )

        return TrainingArtifacts(
            series=series,
            temperature_series=temperature_series,
            scaler_y=scaler_y,
            scaler_temperature=scaler_temperature,
            model=model,
            dependencies=self.dependencies,
        )


class ForecastGenerator:
    def __init__(
        self,
        settings: NHitsSettings,
        runtime: RuntimeConfiguration,
    ) -> None:
        self.settings = settings
        self.runtime = runtime

    def forecast_last_week(self, artifacts: TrainingArtifacts, calibrate: bool = True) -> pd.DataFrame:
        time_index = artifacts.series.time_index
        if len(time_index) < self.settings.test_horizon + 1:
            raise NHitsError("insufficient time series length for test forecast")

        intercept = 0.0
        slope = 1.0

        calibration_ready = len(time_index) >= (
            self.settings.test_horizon + self.settings.calibration_horizon + 1
        )
        if calibrate and calibration_ready:
            calibration_anchor = time_index[
                -(self.settings.test_horizon + self.settings.calibration_horizon + 1)
            ]
            calibration_true = artifacts.series[
                -(self.settings.test_horizon + self.settings.calibration_horizon) : -self.settings.test_horizon
            ]
            calibration_pred = self._predict_window(
                artifacts,
                calibration_anchor,
                self.settings.calibration_horizon,
            ).slice_intersect(calibration_true)
            intercept, slope = self._fit_global_bias(calibration_true, calibration_pred)

        test_anchor = time_index[-(self.settings.test_horizon + 1)]
        test_true = artifacts.series[-self.settings.test_horizon :]
        test_pred = self._predict_window(
            artifacts,
            test_anchor,
            self.settings.test_horizon,
        ).slice_intersect(test_true)

        forecast_frame = pd.DataFrame(
            {
                "start_dt": test_true.time_index,
                "y": test_true.values().flatten(),
                "y_hat": test_pred.values().flatten(),
            }
        )

        if calibrate:
            forecast_frame["y_hat"] = intercept + slope * forecast_frame["y_hat"].astype(float)

        return forecast_frame

    def _predict_window(
        self,
        artifacts: TrainingArtifacts,
        anchor_end_ts: pd.Timestamp,
        horizon: int,
    ) -> Any:
        series_input = artifacts.series[:anchor_end_ts]
        temperature_input = None

        if artifacts.temperature_series is not None:
            required_covariate_end = self._required_covariate_end(anchor_end_ts, horizon)
            temperature_cut = artifacts.temperature_series[:anchor_end_ts]
            temperature_input = self._extend_covariate_to(
                temperature_cut,
                required_covariate_end,
                artifacts.dependencies.TimeSeries,
            )

        series_scaled = artifacts.scaler_y.transform(series_input)
        if temperature_input is not None and artifacts.scaler_temperature is not None:
            temperature_scaled = artifacts.scaler_temperature.transform(temperature_input)
        else:
            temperature_scaled = None

        prediction = artifacts.model.predict(
            n=horizon,
            series=series_scaled,
            past_covariates=temperature_scaled,
            show_warnings=False,
            dataloader_kwargs=self.runtime.dataloader_kwargs,
        )
        return artifacts.scaler_y.inverse_transform(prediction)

    def _required_covariate_end(self, anchor_end_ts: pd.Timestamp, horizon: int) -> pd.Timestamp:
        return anchor_end_ts + pd.Timedelta(
            hours=max(0, horizon - self.settings.output_chunk_length)
        )

    @staticmethod
    def _extend_covariate_to(series: Any, new_end_ts: pd.Timestamp, time_series_type: Any) -> Any:
        last_timestamp = series.time_index[-1]
        if last_timestamp >= new_end_ts:
            return series[:new_end_ts]

        full_times = pd.date_range(start=series.time_index[0], end=new_end_ts, freq="H")
        values = series.values(copy=True)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        current_length = values.shape[0]
        required_length = len(full_times)

        if required_length > current_length:
            last_row = values[-1:].astype(np.float32)
            padding = np.repeat(last_row, required_length - current_length, axis=0)
            extended_values = np.vstack([values, padding]).astype(np.float32)
        else:
            extended_values = values[:required_length].astype(np.float32)

        return time_series_type.from_times_and_values(
            full_times,
            extended_values,
            columns=list(series.components),
            freq="H",
        ).astype(np.float32)

    @staticmethod
    def _fit_global_bias(y_true: Any, y_pred: Any) -> tuple[float, float]:
        target_values = y_true.values().flatten()
        predicted_values = y_pred.values().flatten()
        design_matrix = np.vstack([np.ones_like(predicted_values), predicted_values]).T
        solution, *_ = np.linalg.lstsq(design_matrix, target_values, rcond=None)
        intercept = float(solution[0])
        slope = float(np.clip(solution[1], 0.85, 1.15))
        return intercept, slope


class MetricsCalculator:
    def __init__(self, dependencies: DartsDependencies) -> None:
        self.dependencies = dependencies

    def build(self, forecast_frame: pd.DataFrame) -> pd.DataFrame:
        TimeSeries = self.dependencies.TimeSeries
        true_series = TimeSeries.from_dataframe(
            forecast_frame,
            time_col="start_dt",
            value_cols="y",
            freq="H",
        )
        predicted_series = TimeSeries.from_dataframe(
            forecast_frame,
            time_col="start_dt",
            value_cols="y_hat",
            freq="H",
        )

        rows = []
        for label, horizon in (("24h", 24), ("72h", 72), ("168h", 168)):
            series_true = true_series[-horizon:]
            series_pred = predicted_series[-horizon:]
            rows.append(
                {
                    "horizon": label,
                    "MAE": float(self.dependencies.mae(series_true, series_pred)),
                    "RMSE": float(self.dependencies.rmse(series_true, series_pred)),
                    "MAPE_%": float(self.dependencies.mape(series_true, series_pred)),
                    "R2": float(self.dependencies.r2_score(series_true, series_pred)),
                }
            )

        return pd.DataFrame(rows)


class ForecastPlotter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def save(self, forecast_frame: pd.DataFrame) -> tuple[Path, ...]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        figure_paths = (
            self._save_window(
                forecast_frame.tail(168),
                "Прогноз на 1 неделю",
                "nhits_stable_week_168h.png",
            ),
            self._save_window(
                forecast_frame.tail(72),
                "Прогноз на 3 суток",
                "nhits_stable_3days_72h.png",
            ),
            self._save_window(
                forecast_frame.tail(24),
                "Прогноз на 1 сутки",
                "nhits_stable_1day_24h.png",
            ),
        )
        return figure_paths

    def _save_window(
        self,
        window_frame: pd.DataFrame,
        title: str,
        file_name: str,
    ) -> Path:
        figure, axis = ForecastFigureStyle.create_figure()
        deviation = self._percent_deviation(window_frame["y"], window_frame["y_hat"])
        deviation_text = f"{deviation:.2f}%" if np.isfinite(deviation) else "н/д"

        axis.plot(
            window_frame["start_dt"],
            window_frame["y"],
            label="Факт",
            linewidth=2.2,
            color=ForecastFigureStyle.actual_color,
        )
        axis.plot(
            window_frame["start_dt"],
            window_frame["y_hat"],
            label="N-HiTS",
            linewidth=2.2,
            color=ForecastFigureStyle.model_color,
        )

        ForecastFigureStyle.style_axis(axis, f"{title} | отклонение: {deviation_text}")
        axis.legend(loc="upper left", ncol=2, handlelength=3)

        figure.tight_layout()
        output_path = self.output_dir / file_name
        figure.savefig(output_path, dpi=ForecastFigureStyle.dpi)
        plt.close(figure)
        return output_path

    @staticmethod
    def _percent_deviation(actual: pd.Series, predicted: pd.Series) -> float:
        actual_values = actual.to_numpy(dtype=float)
        predicted_values = predicted.to_numpy(dtype=float)
        valid_mask = (
            np.isfinite(actual_values)
            & np.isfinite(predicted_values)
            & (np.abs(actual_values) > 1e-12)
        )

        if valid_mask.sum() >= max(3, int(0.5 * len(actual_values))):
            return float(
                np.mean(
                    np.abs((predicted_values[valid_mask] - actual_values[valid_mask]) / actual_values[valid_mask])
                )
                * 100.0
            )

        valid_mask = np.isfinite(actual_values) & np.isfinite(predicted_values)
        actual_values = actual_values[valid_mask]
        predicted_values = predicted_values[valid_mask]
        denominator = np.abs(actual_values).sum()

        if denominator <= 1e-12:
            return float("nan")

        return float(np.abs(predicted_values - actual_values).sum() / denominator * 100.0)


class NHitsPipeline:
    def __init__(
        self,
        project_root: Path,
        output_dir: Path,
        settings: NHitsSettings | None = None,
    ) -> None:
        self.project_root = project_root
        self.output_dir = output_dir
        self.settings = settings or NHitsSettings()
        self.dependencies = DependencyLoader.load()
        self.runtime = RuntimeResolver(self.dependencies).resolve()
        self.locator = DataLocator(project_root)
        self.loader = SourceDataLoader(self.locator)
        self.series_builder = TimeSeriesBuilder(self.dependencies, self.settings)
        self.trainer = NHitsTrainer(self.dependencies, self.settings, self.runtime)
        self.forecaster = ForecastGenerator(self.settings, self.runtime)
        self.metrics_calculator = MetricsCalculator(self.dependencies)
        self.plotter = ForecastPlotter(output_dir)

    def run(self) -> RunArtifacts:
        PlotStyle.apply()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._configure_runtime()

        source_frame = self.loader.load()
        series, temperature_series = self.series_builder.build(source_frame)
        training_artifacts = self.trainer.fit(series, temperature_series)
        forecast_frame = self.forecaster.forecast_last_week(training_artifacts, calibrate=True)
        metrics_frame = self.metrics_calculator.build(forecast_frame)

        forecast_csv_path = self._save_forecast(forecast_frame)
        metrics_csv_path = self._save_metrics(metrics_frame)
        figure_paths = self.plotter.save(forecast_frame)

        self._release_resources(training_artifacts)

        return RunArtifacts(
            output_dir=self.output_dir,
            forecast_csv_path=forecast_csv_path,
            metrics_csv_path=metrics_csv_path,
            figure_paths=figure_paths,
        )

    def _configure_runtime(self) -> None:
        torch = self.dependencies.torch
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("medium")
            torch.backends.cudnn.benchmark = True

    def _save_forecast(self, forecast_frame: pd.DataFrame) -> Path:
        forecast_frame.to_csv(OUTPUT_CSV, index=False)
        return OUTPUT_CSV

    def _save_metrics(self, metrics_frame: pd.DataFrame) -> Path:
        metrics_frame.to_csv(METRICS_CSV, index=False)
        return METRICS_CSV

    def _release_resources(self, training_artifacts: TrainingArtifacts) -> None:
        torch = self.dependencies.torch
        del training_artifacts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def run_nhits(output_dir: Path = OUTPUT_DIR) -> bool:
    pipeline = NHitsPipeline(PROJECT_ROOT, Path(output_dir))
    pipeline.run()
    return True


def main() -> bool:
    return run_nhits()


def cli() -> int:
    try:
        return 0 if main() else 1
    except NHitsError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(cli())
