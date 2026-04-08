from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

try:
    from .DataIO import read_tabular_frame
except ImportError:
    from DataIO import read_tabular_frame


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Photo" / "article_feature_ranking_plots"


class CatBoostError(RuntimeError):
    pass


class DependencyError(CatBoostError):
    pass


@dataclass(frozen=True, slots=True)
class CatBoostSettings:
    input_chunk_length: int = 720
    output_chunk_length: int = 24
    y_lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24, 48, 72, 168)
    temperature_lags: tuple[int, ...] = (1, 6, 24, 168)
    rolling_windows: tuple[int, ...] = (6, 24, 168)
    iterations: int = 2000
    depth: int = 8
    learning_rate: float = 0.03
    l2_leaf_reg: float = 3.0
    early_stopping_rounds: int = 100
    random_state: int = 42

    @property
    def validation_horizon(self) -> int:
        return max(self.input_chunk_length + self.output_chunk_length, 360)


@dataclass(frozen=True, slots=True)
class DataColumns:
    time: str
    consumption: str
    temperature: str | None


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    feature_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    output_dir: Path
    importance_csv_path: Path
    grouped_importance_csv_path: Path
    figure_paths: tuple[Path, ...]


class PlotStyle:
    @staticmethod
    def apply() -> None:
        plt.rcParams.update(
            {
                "font.family": "DejaVu Serif",
                "font.size": 11,
                "figure.facecolor": "#FFFFFF",
                "axes.facecolor": "#FFFFFF",
                "axes.edgecolor": "#3A3A3A",
                "axes.labelcolor": "#1F1F1F",
                "xtick.color": "#1F1F1F",
                "ytick.color": "#1F1F1F",
                "text.color": "#1F1F1F",
                "legend.frameon": False,
                "savefig.facecolor": "#FFFFFF",
            }
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

        raise CatBoostError("source data file not found")

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
        result["start_dt"] = pd.to_datetime(result[columns.time], dayfirst=True, errors="coerce")
        result["y"] = result[columns.consumption].map(self._coerce_numeric)

        if columns.temperature is not None:
            result["temperature"] = result[columns.temperature].map(self._coerce_numeric)
        else:
            result["temperature"] = np.nan

        result = result.loc[result["start_dt"].notna() & result["y"].notna()].copy()
        result["start_dt"] = result["start_dt"].dt.floor("h")
        result = result.sort_values("start_dt").reset_index(drop=True)

        if result.duplicated(subset="start_dt").any():
            result = (
                result.groupby("start_dt", as_index=False)
                .agg(y=("y", "mean"), temperature=("temperature", "mean"))
                .sort_values("start_dt")
                .reset_index(drop=True)
            )

        if result.empty:
            raise CatBoostError("source dataset is empty after preprocessing")

        return result[["start_dt", "y", "temperature"]]

    @staticmethod
    def _read_frame(source_path: Path) -> pd.DataFrame:
        return read_tabular_frame(source_path)

    def _resolve_columns(self, frame: pd.DataFrame) -> DataColumns:
        column_names = frame.columns.tolist()
        time_column = self._find_column(column_names, ("начало", "start")) or column_names[0]
        consumption_column = self._find_column(column_names, ("потреб", "consum", "load"))
        temperature_column = self._find_column(column_names, ("темпер", "temp"))

        if consumption_column is None:
            raise CatBoostError("consumption column not found")

        return DataColumns(
            time=time_column,
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
            return float(value)

        text = str(value).strip()
        if not text:
            return float("nan")

        text = text.replace(" ", "").replace(",", ".")
        try:
            return float(text)
        except ValueError:
            return float("nan")


class FeatureBuilder:
    def __init__(self, settings: CatBoostSettings) -> None:
        self.settings = settings

    def build(self, source_frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
        frame = source_frame.copy()

        for lag in self.settings.y_lags:
            frame[f"y_lag_{lag}"] = frame["y"].shift(lag)

        if frame["temperature"].notna().any():
            for lag in self.settings.temperature_lags:
                frame[f"temp_lag_{lag}"] = frame["temperature"].shift(lag)

        for window in self.settings.rolling_windows:
            frame[f"y_roll_mean_{window}"] = (
                frame["y"].shift(1).rolling(window=window, min_periods=max(3, window // 2)).mean()
            )

        frame = self._add_calendar_features(frame)
        feature_columns = tuple(
            column for column in frame.columns if column not in {"start_dt", "y", "temperature"}
        )

        if not feature_columns:
            raise CatBoostError("feature set is empty")

        frame = frame.dropna(subset=[*feature_columns, "y"]).reset_index(drop=True)
        if frame.empty:
            raise CatBoostError("feature table is empty after preprocessing")

        return frame, feature_columns

    @staticmethod
    def _add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        index = pd.DatetimeIndex(result["start_dt"])

        result["hour"] = index.hour
        result["dow"] = index.dayofweek
        result["month"] = index.month

        result["hour_sin"], result["hour_cos"] = FeatureBuilder._encode_cycle(result["hour"], 24)
        result["dow_sin"], result["dow_cos"] = FeatureBuilder._encode_cycle(result["dow"], 7)
        result["month_sin"], result["month_cos"] = FeatureBuilder._encode_cycle(result["month"], 12)
        result["is_weekend"] = (result["dow"] >= 5).astype(int)
        return result

    @staticmethod
    def _encode_cycle(values: pd.Series, period: int) -> tuple[np.ndarray, np.ndarray]:
        numeric_values = values.astype(float).to_numpy()
        radians = 2 * np.pi * numeric_values / float(period)
        return np.sin(radians), np.cos(radians)


class ModelTrainer:
    def __init__(self, settings: CatBoostSettings) -> None:
        self.settings = settings

    def fit(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        feature_columns: tuple[str, ...],
    ) -> CatBoostRegressor:
        CatBoostRegressor, Pool = load_catboost_dependencies()
        model = CatBoostRegressor(
            iterations=self.settings.iterations,
            depth=self.settings.depth,
            learning_rate=self.settings.learning_rate,
            l2_leaf_reg=self.settings.l2_leaf_reg,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=self.settings.random_state,
            od_type="Iter",
            od_wait=self.settings.early_stopping_rounds,
            use_best_model=True,
            allow_writing_files=False,
            verbose=False,
        )

        train_pool = Pool(train_frame[list(feature_columns)], train_frame["y"])
        validation_pool = Pool(validation_frame[list(feature_columns)], validation_frame["y"])
        model.fit(train_pool, eval_set=validation_pool)
        return model


class FeatureNameFormatter:
    @staticmethod
    def to_label(feature_name: str) -> str:
        y_lag_match = re.fullmatch(r"y_lag_(\d+)", feature_name)
        if y_lag_match is not None:
            return f"Потребление: лаг {y_lag_match.group(1)} ч"

        temperature_lag_match = re.fullmatch(r"temp_lag_(\d+)", feature_name)
        if temperature_lag_match is not None:
            return f"Температура: лаг {temperature_lag_match.group(1)} ч"

        rolling_match = re.fullmatch(r"y_roll_mean_(\d+)", feature_name)
        if rolling_match is not None:
            return f"Потребление: скользящее среднее {rolling_match.group(1)} ч"

        mapping = {
            "hour": "Час",
            "dow": "День недели",
            "month": "Месяц",
            "hour_sin": "Час (sin)",
            "hour_cos": "Час (cos)",
            "dow_sin": "День недели (sin)",
            "dow_cos": "День недели (cos)",
            "month_sin": "Месяц (sin)",
            "month_cos": "Месяц (cos)",
            "is_weekend": "Выходной день",
        }
        return mapping.get(feature_name, feature_name)

    @staticmethod
    def to_group(feature_name: str) -> str:
        if feature_name.startswith("y_lag_"):
            return "Лаги потребления"
        if feature_name.startswith("temp_lag_"):
            return "Температура"
        if feature_name.startswith("y_roll_mean_"):
            return "Скользящие средние"
        if feature_name in {
            "hour",
            "dow",
            "month",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "is_weekend",
        }:
            return "Календарные признаки"
        return "Прочее"


class ImportanceTableBuilder:
    def build(
        self,
        model: CatBoostRegressor,
        reference_frame: pd.DataFrame,
        feature_columns: tuple[str, ...],
    ) -> pd.DataFrame:
        _, Pool = load_catboost_dependencies()
        reference_pool = Pool(reference_frame[list(feature_columns)])
        importance_values = model.get_feature_importance(
            reference_pool,
            type="PredictionValuesChange",
        )

        importance_frame = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": importance_values,
            }
        )

        total_importance = float(importance_frame["importance"].sum())
        if total_importance > 0:
            importance_frame["importance_pct"] = (
                100.0 * importance_frame["importance"] / total_importance
            )
        else:
            importance_frame["importance_pct"] = 0.0

        importance_frame["feature_label"] = importance_frame["feature"].map(
            FeatureNameFormatter.to_label
        )
        importance_frame["group"] = importance_frame["feature"].map(FeatureNameFormatter.to_group)

        return (
            importance_frame.sort_values("importance_pct", ascending=False)
            .reset_index(drop=True)
        )

    @staticmethod
    def build_grouped_table(importance_frame: pd.DataFrame) -> pd.DataFrame:
        return (
            importance_frame.groupby("group", as_index=False)["importance_pct"]
            .sum()
            .sort_values("importance_pct", ascending=False)
            .reset_index(drop=True)
        )


class ImportancePlotter:
    primary_color = "#0D53E0"
    text_color = "#1F1F1F"

    def save_top_features(
        self,
        importance_frame: pd.DataFrame,
        output_path: Path,
        top_n: int = 25,
    ) -> Path:
        top_frame = importance_frame.head(top_n).iloc[::-1].copy()
        figure, axis = plt.subplots(figsize=(10, 8), dpi=150)

        bars = axis.barh(
            top_frame["feature_label"],
            top_frame["importance_pct"],
            color=self.primary_color,
            edgecolor="#2A2A2A",
            linewidth=0.6,
            height=0.72,
        )

        self._format_axis(axis, "Вклад в модель, %")
        axis.xaxis.set_major_formatter(PercentFormatter(100))
        axis.set_title("Ранжирование факторов", pad=14)

        self._add_bar_labels(axis, bars)
        axis.set_xlim(0.0, self._axis_limit(float(top_frame["importance_pct"].max())))

        figure.tight_layout()
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def save_grouped_importance(
        self,
        grouped_frame: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        ordered_frame = grouped_frame.sort_values("importance_pct", ascending=True).copy()
        figure, axis = plt.subplots(figsize=(10, 5), dpi=150)

        bars = axis.barh(
            ordered_frame["group"],
            ordered_frame["importance_pct"],
            color=self.primary_color,
            edgecolor="#2A2A2A",
            linewidth=0.6,
            height=0.64,
        )

        self._format_axis(axis, "Вклад в модель, %")
        axis.xaxis.set_major_formatter(PercentFormatter(100))
        axis.set_title("Суммарный вклад групп факторов", pad=14)

        self._add_bar_labels(axis, bars, fontweight="bold")
        axis.set_xlim(0.0, self._axis_limit(float(ordered_frame["importance_pct"].max())))

        figure.tight_layout()
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def save_key_predictors(
        self,
        importance_frame: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        summary_frame = self._build_key_predictors_frame(importance_frame)
        figure, axis = plt.subplots(figsize=(10, 6), dpi=150)

        bars = axis.barh(
            summary_frame["factor"],
            summary_frame["importance_pct"],
            color=self.primary_color,
            edgecolor="#2A2A2A",
            linewidth=0.6,
            height=0.66,
        )

        self._format_axis(axis, "Вклад в модель, %")
        axis.xaxis.set_major_formatter(PercentFormatter(100))
        axis.set_title("Ключевые предикторы", pad=14)

        self._add_bar_labels(axis, bars)
        axis.set_xlim(0.0, self._axis_limit(float(summary_frame["importance_pct"].max())))

        figure.tight_layout()
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def _format_axis(self, axis: plt.Axes, x_label: str) -> None:
        axis.set_axisbelow(True)
        axis.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.4, color="#8E8E8E")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlabel(x_label)
        axis.tick_params(axis="y", labelsize=10)

    def _add_bar_labels(
        self,
        axis: plt.Axes,
        bars,
        fontweight: str = "normal",
    ) -> None:
        max_value = max((float(bar.get_width()) for bar in bars), default=0.0)
        offset = 0.01 * max_value if max_value > 0 else 0.1
        for bar in bars:
            width = float(bar.get_width())
            axis.text(
                width + offset,
                bar.get_y() + bar.get_height() / 2.0,
                f"{width:.1f}%",
                va="center",
                ha="left",
                fontsize=9,
                fontweight=fontweight,
                color=self.text_color,
            )

    @staticmethod
    def _axis_limit(max_value: float) -> float:
        return max_value * 1.15 if max_value > 0 else 1.0

    def _build_key_predictors_frame(self, importance_frame: pd.DataFrame) -> pd.DataFrame:
        lag_values = importance_frame["feature"].map(self._extract_y_lag)

        short_lags = float(
            importance_frame.loc[
                lag_values.notna() & (lag_values.astype(float) <= 12.0),
                "importance_pct",
            ].sum()
        )
        lag_24 = float(
            importance_frame.loc[
                importance_frame["feature"] == "y_lag_24",
                "importance_pct",
            ].sum()
        )
        lag_168 = float(
            importance_frame.loc[
                importance_frame["feature"] == "y_lag_168",
                "importance_pct",
            ].sum()
        )
        temperature = float(
            importance_frame.loc[
                importance_frame["feature"].str.startswith("temp_lag_"),
                "importance_pct",
            ].sum()
        )
        calendar = float(
            importance_frame.loc[
                importance_frame["group"] == "Календарные признаки",
                "importance_pct",
            ].sum()
        )
        rolling = float(
            importance_frame.loc[
                importance_frame["feature"].str.startswith("y_roll_mean_"),
                "importance_pct",
            ].sum()
        )

        summary_frame = pd.DataFrame(
            {
                "factor": [
                    "Короткие лаги 1-12 ч",
                    "Лаг 24 ч",
                    "Лаг 168 ч",
                    "Температура",
                    "Календарные признаки",
                    "Скользящие средние",
                ],
                "importance_pct": [
                    short_lags,
                    lag_24,
                    lag_168,
                    temperature,
                    calendar,
                    rolling,
                ],
            }
        )

        return summary_frame.sort_values("importance_pct", ascending=True).reset_index(drop=True)

    @staticmethod
    def _extract_y_lag(feature_name: str) -> int | None:
        match = re.fullmatch(r"y_lag_(\d+)", feature_name)
        return int(match.group(1)) if match is not None else None


class CatBoostFeatureImportancePipeline:
    def __init__(
        self,
        project_root: Path,
        output_dir: Path,
        settings: CatBoostSettings | None = None,
    ) -> None:
        self.project_root = project_root
        self.output_dir = output_dir
        self.settings = settings or CatBoostSettings()
        self.locator = DataLocator(project_root)
        self.loader = SourceDataLoader(self.locator)
        self.feature_builder = FeatureBuilder(self.settings)
        self.trainer = ModelTrainer(self.settings)
        self.importance_builder = ImportanceTableBuilder()
        self.plotter = ImportancePlotter()

    def run(self) -> RunArtifacts:
        PlotStyle.apply()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        source_frame = self.loader.load()
        supervised_frame, feature_columns = self.feature_builder.build(source_frame)
        split = self._split_dataset(supervised_frame, feature_columns)
        model = self.trainer.fit(
            split.train_frame,
            split.validation_frame,
            split.feature_columns,
        )

        importance_frame = self.importance_builder.build(
            model,
            split.validation_frame,
            split.feature_columns,
        )
        grouped_frame = self.importance_builder.build_grouped_table(importance_frame)

        importance_csv_path = self._save_importance_table(importance_frame)
        grouped_importance_csv_path = self._save_grouped_table(grouped_frame)

        figure_paths = (
            self.plotter.save_top_features(
                importance_frame,
                self.output_dir / "01_top25_features.png",
                top_n=25,
            ),
            self.plotter.save_grouped_importance(
                grouped_frame,
                self.output_dir / "02_groups_importance.png",
            ),
            self.plotter.save_key_predictors(
                importance_frame,
                self.output_dir / "03_key_factors_6bars.png",
            ),
        )

        return RunArtifacts(
            output_dir=self.output_dir,
            importance_csv_path=importance_csv_path,
            grouped_importance_csv_path=grouped_importance_csv_path,
            figure_paths=figure_paths,
        )

    def _split_dataset(
        self,
        supervised_frame: pd.DataFrame,
        feature_columns: tuple[str, ...],
    ) -> DatasetSplit:
        minimum_length = self.settings.validation_horizon + 100
        if len(supervised_frame) < minimum_length:
            raise CatBoostError("insufficient data after feature engineering")

        split_timestamp = supervised_frame["start_dt"].iloc[-self.settings.validation_horizon]
        train_frame = supervised_frame.loc[supervised_frame["start_dt"] < split_timestamp].copy()
        validation_frame = supervised_frame.loc[
            supervised_frame["start_dt"] >= split_timestamp
        ].copy()

        if train_frame.empty or validation_frame.empty:
            raise CatBoostError("invalid training or validation split")

        return DatasetSplit(
            train_frame=train_frame,
            validation_frame=validation_frame,
            feature_columns=feature_columns,
        )

    def _save_importance_table(self, importance_frame: pd.DataFrame) -> Path:
        output_path = self.output_dir / "feature_importance_full.csv"
        importance_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path

    def _save_grouped_table(self, grouped_frame: pd.DataFrame) -> Path:
        output_path = self.output_dir / "feature_importance_groups.csv"
        grouped_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path


def load_catboost_dependencies():
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as exc:
        raise DependencyError("catboost package is not installed") from exc
    return CatBoostRegressor, Pool


def run_feature_importance(output_dir: Path = OUTPUT_DIR) -> bool:
    pipeline = CatBoostFeatureImportancePipeline(PROJECT_ROOT, Path(output_dir))
    pipeline.run()
    return True


def main() -> bool:
    return run_feature_importance()


def cli() -> int:
    try:
        return 0 if main() else 1
    except CatBoostError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(cli())
