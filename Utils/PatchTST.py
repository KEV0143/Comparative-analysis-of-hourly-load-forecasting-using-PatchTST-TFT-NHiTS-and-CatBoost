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
OUTPUT_DIR = PROJECT_ROOT / "Photo" / "plots_patchtst_gluonts"
OUTPUT_CSV = OUTPUT_DIR / "predictions_last_week_patchtst.csv"
METRICS_CSV = OUTPUT_DIR / "metrics_by_horizon_patchtst.csv"


class PatchTSTError(RuntimeError):
    pass


class DependencyError(PatchTSTError):
    pass


@dataclass(frozen=True, slots=True)
class PatchTSTSettings:
    input_chunk_length: int = 720
    output_chunk_length: int = 24
    test_horizon: int = 168
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_size: int = 32
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.15
    patch_len: int = 24
    patch_stride: int = 8
    random_state: int = 42
    freq: str = "H"


@dataclass(frozen=True, slots=True)
class DataColumns:
    start: str
    end: str | None
    consumption: str
    temperature: str | None


@dataclass(frozen=True, slots=True)
class RuntimeConfiguration:
    trainer_kwargs: dict[str, Any]
    num_workers: int


@dataclass(frozen=True, slots=True)
class PatchTSTDependencies:
    torch: Any
    ListDataset: Any
    PatchTSTEstimator: Any
    StudentTOutput: Any


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    start: pd.Timestamp
    freq: str
    target_train: np.ndarray
    temperature_test_full: np.ndarray
    predictor: Any


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
    def load() -> PatchTSTDependencies:
        try:
            import torch
            from gluonts.dataset.common import ListDataset
            from gluonts.torch.distributions.studentT import StudentTOutput
            from gluonts.torch.model.patch_tst import PatchTSTEstimator
        except ImportError as exc:
            raise DependencyError("patchtst dependencies are unavailable") from exc

        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        return PatchTSTDependencies(
            torch=torch,
            ListDataset=ListDataset,
            PatchTSTEstimator=PatchTSTEstimator,
            StudentTOutput=StudentTOutput,
        )


class RuntimeResolver:
    def __init__(self, dependencies: PatchTSTDependencies, project_root: Path, settings: PatchTSTSettings) -> None:
        self.dependencies = dependencies
        self.project_root = project_root
        self.settings = settings

    def resolve(self) -> RuntimeConfiguration:
        torch = self.dependencies.torch
        use_gpu = bool(torch.cuda.is_available())

        trainer_kwargs = {
            "accelerator": "gpu" if use_gpu else "cpu",
            "devices": 1,
            "precision": 16 if use_gpu else 32,
            "enable_model_summary": False,
            "log_every_n_steps": 50,
            "num_sanity_val_steps": 0,
            "deterministic": False,
            "max_epochs": self.settings.epochs,
            "logger": False,
            "enable_progress_bar": False,
            "default_root_dir": str(self.project_root / "lightning_logs_patchtst"),
        }

        if use_gpu:
            torch.set_float32_matmul_precision("medium")
            torch.backends.cudnn.benchmark = True

        is_windows = (os.name == "nt") or ("windows" in platform.system().lower())
        cpu_total = os.cpu_count() or 2
        num_workers = 0 if is_windows else max(1, min(4, cpu_total - 1))
        return RuntimeConfiguration(trainer_kwargs=trainer_kwargs, num_workers=num_workers)


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

        raise PatchTSTError("source data file not found")

    @staticmethod
    def _first_existing_path(candidates: tuple[Path, ...]) -> Path | None:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None


class SourceDataLoader:
    def __init__(self, locator: DataLocator, settings: PatchTSTSettings) -> None:
        self.locator = locator
        self.settings = settings

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
            raise PatchTSTError("source dataset is empty after preprocessing")

        result = self._regularize_frame(result)
        if result.empty:
            raise PatchTSTError("source dataset is empty after regularization")

        return result[["start_dt", "end_dt", "consumption_mwh", "temperature_c"]]

    @staticmethod
    def _read_frame(source_path: Path) -> pd.DataFrame:
        return read_tabular_frame(source_path)

    def _resolve_columns(self, frame: pd.DataFrame) -> DataColumns:
        column_names = frame.columns.tolist()
        start_column = self._find_column(column_names, ("начало", "start")) or column_names[0]
        end_column = self._find_column(column_names, ("конец", "end"))
        consumption_column = self._find_column(column_names, ("потреб", "consum", "load"))
        temperature_column = self._find_column(column_names, ("темпер", "temp"))

        if consumption_column is None:
            raise PatchTSTError("consumption column not found")

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

    def _regularize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        regularized = (
            frame.set_index("start_dt")
            .reindex(
                pd.date_range(
                    frame["start_dt"].min(),
                    frame["start_dt"].max(),
                    freq=self.settings.freq.lower(),
                )
            )
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
        else:
            regularized["temperature_c"] = 0.0

        first_target_index = regularized["consumption_mwh"].first_valid_index()
        if first_target_index is None:
            return regularized.iloc[0:0].copy()

        regularized = regularized.loc[first_target_index:].reset_index(drop=True)
        return regularized


class PatchTSTTrainer:
    def __init__(
        self,
        dependencies: PatchTSTDependencies,
        settings: PatchTSTSettings,
        runtime: RuntimeConfiguration,
    ) -> None:
        self.dependencies = dependencies
        self.settings = settings
        self.runtime = runtime

    def fit(self, source_frame: pd.DataFrame) -> TrainingArtifacts:
        torch = self.dependencies.torch
        torch.manual_seed(self.settings.random_state)
        np.random.seed(self.settings.random_state)

        start_timestamp = pd.Timestamp(source_frame["start_dt"].min())
        target_values = source_frame["consumption_mwh"].astype(float).to_numpy()
        temperature_values = source_frame["temperature_c"].astype(float).to_numpy()

        if len(target_values) < self.settings.test_horizon + 1:
            raise PatchTSTError("insufficient time series length for test forecast")

        anchor_end_index = len(target_values) - (self.settings.test_horizon + 1)
        target_train = target_values[: anchor_end_index + 1]
        temperature_history = temperature_values[: anchor_end_index + 1]
        temperature_test_full = self._extend_locf(temperature_history, self.settings.test_horizon)

        train_dataset = self.dependencies.ListDataset(
            [
                {
                    "start": start_timestamp,
                    "target": target_train.astype(np.float32),
                    "feat_dynamic_real": [temperature_history.astype(np.float32)],
                }
            ],
            freq=self.settings.freq,
        )

        estimator = self.dependencies.PatchTSTEstimator(
            prediction_length=self.settings.test_horizon,
            context_length=self.settings.input_chunk_length,
            patch_len=self.settings.patch_len,
            stride=self.settings.patch_stride,
            d_model=self.settings.hidden_size,
            nhead=self.settings.n_heads,
            num_encoder_layers=self.settings.n_layers,
            dim_feedforward=4 * self.settings.hidden_size,
            dropout=self.settings.dropout,
            lr=self.settings.learning_rate,
            batch_size=self.settings.batch_size,
            trainer_kwargs=self.runtime.trainer_kwargs,
            distr_output=self.dependencies.StudentTOutput(),
            scaling="mean",
        )

        predictor = estimator.train(training_data=train_dataset)
        return TrainingArtifacts(
            start=start_timestamp,
            freq=self.settings.freq,
            target_train=target_train,
            temperature_test_full=temperature_test_full,
            predictor=predictor,
        )

    @staticmethod
    def _extend_locf(values: np.ndarray, additional_length: int) -> np.ndarray:
        if additional_length <= 0:
            return values.copy()
        last_value = values[-1]
        padding = np.repeat(last_value, additional_length)
        return np.concatenate([values, padding])


class ForecastGenerator:
    def __init__(self, dependencies: PatchTSTDependencies, settings: PatchTSTSettings) -> None:
        self.dependencies = dependencies
        self.settings = settings

    def forecast_last_week(
        self,
        artifacts: TrainingArtifacts,
        source_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        test_dataset = self.dependencies.ListDataset(
            [
                {
                    "start": artifacts.start,
                    "target": artifacts.target_train.astype(np.float32),
                    "feat_dynamic_real": [artifacts.temperature_test_full.astype(np.float32)],
                }
            ],
            freq=artifacts.freq,
        )

        forecasts = list(artifacts.predictor.predict(test_dataset))
        forecast = forecasts[0]
        predicted_values = forecast.mean.astype(float)
        actual_values = source_frame["consumption_mwh"].to_numpy(dtype=float)[-self.settings.test_horizon :]
        actual_index = source_frame["start_dt"].iloc[-self.settings.test_horizon :].to_numpy()

        return pd.DataFrame(
            {
                "start_dt": actual_index,
                "y": actual_values,
                "y_hat": predicted_values,
            }
        )


class MetricsCalculator:
    def build(self, forecast_frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for label, horizon in (("24h", 24), ("72h", 72), ("168h", 168)):
            window_frame = forecast_frame.tail(horizon)
            rows.append(
                {
                    "horizon": label,
                    "MAE": self._mae(window_frame["y"], window_frame["y_hat"]),
                    "RMSE": self._rmse(window_frame["y"], window_frame["y_hat"]),
                    "MAPE_%": self._mape(window_frame["y"], window_frame["y_hat"]),
                    "R2": self._r2(window_frame["y"], window_frame["y_hat"]),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _mape(actual: pd.Series, predicted: pd.Series) -> float:
        actual_values = np.asarray(actual, dtype=float)
        predicted_values = np.asarray(predicted, dtype=float)
        mask = np.isfinite(actual_values) & np.isfinite(predicted_values) & (np.abs(actual_values) > 1e-12)
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(np.abs((predicted_values[mask] - actual_values[mask]) / actual_values[mask])) * 100.0)

    @staticmethod
    def _rmse(actual: pd.Series, predicted: pd.Series) -> float:
        actual_values = np.asarray(actual, dtype=float)
        predicted_values = np.asarray(predicted, dtype=float)
        mask = np.isfinite(actual_values) & np.isfinite(predicted_values)
        if mask.sum() == 0:
            return float("nan")
        return float(np.sqrt(np.mean((predicted_values[mask] - actual_values[mask]) ** 2)))

    @staticmethod
    def _mae(actual: pd.Series, predicted: pd.Series) -> float:
        actual_values = np.asarray(actual, dtype=float)
        predicted_values = np.asarray(predicted, dtype=float)
        mask = np.isfinite(actual_values) & np.isfinite(predicted_values)
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(np.abs(predicted_values[mask] - actual_values[mask])))

    @staticmethod
    def _r2(actual: pd.Series, predicted: pd.Series) -> float:
        actual_values = np.asarray(actual, dtype=float)
        predicted_values = np.asarray(predicted, dtype=float)
        mask = np.isfinite(actual_values) & np.isfinite(predicted_values)
        if mask.sum() < 2:
            return float("nan")
        actual_values = actual_values[mask]
        predicted_values = predicted_values[mask]
        residual_sum = np.sum((actual_values - predicted_values) ** 2)
        total_sum = np.sum((actual_values - np.mean(actual_values)) ** 2)
        if total_sum <= 1e-12:
            return float("nan")
        return float(1.0 - residual_sum / total_sum)


class ForecastPlotter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def save(self, forecast_frame: pd.DataFrame) -> tuple[Path, ...]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        figure_paths = (
            self._save_window(
                forecast_frame.tail(168),
                "Прогноз на 1 неделю",
                "patchtst_week_168h.png",
            ),
            self._save_window(
                forecast_frame.tail(72),
                "Прогноз на 3 суток",
                "patchtst_3days_72h.png",
            ),
            self._save_window(
                forecast_frame.tail(24),
                "Прогноз на 1 сутки",
                "patchtst_1day_24h.png",
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
            label="PatchTST",
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


class PatchTSTPipeline:
    def __init__(
        self,
        project_root: Path,
        output_dir: Path,
        settings: PatchTSTSettings | None = None,
    ) -> None:
        self.project_root = project_root
        self.output_dir = output_dir
        self.settings = settings or PatchTSTSettings()
        self.dependencies = DependencyLoader.load()
        self.runtime = RuntimeResolver(self.dependencies, project_root, self.settings).resolve()
        self.locator = DataLocator(project_root)
        self.loader = SourceDataLoader(self.locator, self.settings)
        self.trainer = PatchTSTTrainer(self.dependencies, self.settings, self.runtime)
        self.forecaster = ForecastGenerator(self.dependencies, self.settings)
        self.metrics_calculator = MetricsCalculator()
        self.plotter = ForecastPlotter(output_dir)

    def run(self) -> RunArtifacts:
        PlotStyle.apply()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        source_frame = self.loader.load()
        training_artifacts = self.trainer.fit(source_frame)
        forecast_frame = self.forecaster.forecast_last_week(training_artifacts, source_frame)
        metrics_frame = self.metrics_calculator.build(forecast_frame)

        forecast_csv_path = self._save_forecast(forecast_frame)
        metrics_csv_path = self._save_metrics(metrics_frame)
        figure_paths = self.plotter.save(forecast_frame)

        self._release_resources()

        return RunArtifacts(
            output_dir=self.output_dir,
            forecast_csv_path=forecast_csv_path,
            metrics_csv_path=metrics_csv_path,
            figure_paths=figure_paths,
        )

    def _save_forecast(self, forecast_frame: pd.DataFrame) -> Path:
        forecast_frame.to_csv(OUTPUT_CSV, index=False)
        return OUTPUT_CSV

    def _save_metrics(self, metrics_frame: pd.DataFrame) -> Path:
        metrics_frame.to_csv(METRICS_CSV, index=False)
        return METRICS_CSV

    def _release_resources(self) -> None:
        torch = self.dependencies.torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def run_patchtst(output_dir: Path = OUTPUT_DIR) -> bool:
    pipeline = PatchTSTPipeline(PROJECT_ROOT, Path(output_dir))
    pipeline.run()
    return True


def main() -> bool:
    return run_patchtst()


def cli() -> int:
    try:
        return 0 if main() else 1
    except PatchTSTError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(cli())
