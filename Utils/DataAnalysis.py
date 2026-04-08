from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from .DataIO import read_tabular_frame
except ImportError:
    from DataIO import read_tabular_frame


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_OUTPUT_DIR = PROJECT_ROOT / "Photo" / "plot_all"
RESIDUALS_OUTPUT_DIR = PROJECT_ROOT / "Photo" / "plots_diagnostics"

PLOT_COLORS = {
    "background": "#FFFFFF",
    "axes_background": "#FFFFFF",
    "grid": "#CFC7BD",
    "text": "#2F2A24",
    "consumption": "#557A95",
    "temperature": "#C97C5D",
    "mean": "#8E3B46",
    "median": "#6A8F5B",
    "mode": "#7A5C99",
    "annotation_box": "#E6D8C9",
}


class DataAnalysisError(RuntimeError):
    pass


class DependencyError(DataAnalysisError):
    pass


@dataclass(frozen=True, slots=True)
class DataColumns:
    time: str
    consumption: str
    temperature: str


@dataclass(frozen=True, slots=True)
class SeriesStatistics:
    count: int
    minimum: float
    maximum: float
    mean: float
    std: float
    median: float
    mode: float
    skewness: float
    kurtosis: float
    q01: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float
    q99: float

    def to_record(self, series_name: str) -> dict[str, float | int | str]:
        return {
            "series": series_name,
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "mode": self.mode,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "q01": self.q01,
            "q05": self.q05,
            "q25": self.q25,
            "q50": self.q50,
            "q75": self.q75,
            "q95": self.q95,
            "q99": self.q99,
        }


@dataclass(frozen=True, slots=True)
class SourceArtifacts:
    source_path: Path
    output_dir: Path
    statistics_path: Path
    overview_path: Path
    figures: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ResidualArtifacts:
    input_path: Path
    output_dir: Path
    summary_path: Path
    ljung_box_path: Path
    figures: tuple[Path, ...]


class DataLocator:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def find_source_data_path(self) -> Path:
        direct_candidates = (
            self.project_root / "Data.xlsx",
            self.project_root / "Data.csv",
            self.project_root / "Data" / "Data.xlsx",
            self.project_root / "Data" / "Data.csv",
        )
        direct_match = self._first_existing_path(direct_candidates)
        if direct_match is not None:
            return direct_match

        for pattern in ("Data.xlsx", "Data.csv"):
            matches = sorted(self.project_root.rglob(pattern))
            if matches:
                return matches[0]

        raise DataAnalysisError("source data file not found")

    def find_catboost_predictions_path(self) -> Path:
        direct_candidates = (
            self.project_root / "Photo" / "plots_catboost" / "predictions_last_week.csv",
            self.project_root / "Photo" / "plots_catboost" / "predictions_last_week_catboost.csv",
            self.project_root / "plots_catboost" / "predictions_last_week.csv",
            self.project_root / "plots_catboost" / "predictions_last_week_catboost.csv",
        )
        direct_match = self._first_existing_path(direct_candidates)
        if direct_match is not None:
            return direct_match

        search_roots = (self.project_root, self.project_root / "Photo")
        for root in search_roots:
            if not root.exists():
                continue
            for candidate in sorted(root.rglob("predictions_last_week*.csv")):
                searchable_name = f"{candidate.parent.name} {candidate.name}".lower()
                if "catboost" in searchable_name or "atboost" in searchable_name:
                    return candidate

        raise DataAnalysisError("catboost prediction file not found")

    @staticmethod
    def _first_existing_path(candidates: Iterable[Path]) -> Path | None:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None


class SourcePlotStyle:
    @staticmethod
    def apply() -> None:
        sns.set_theme(style="white")
        plt.rcParams.update(
            {
                "font.family": "DejaVu Serif",
                "figure.figsize": (14, 9),
                "figure.facecolor": PLOT_COLORS["background"],
                "axes.facecolor": PLOT_COLORS["axes_background"],
                "axes.edgecolor": "#BDB4AA",
                "axes.labelsize": 14,
                "axes.titlesize": 17,
                "axes.titleweight": "bold",
                "axes.labelweight": "bold",
                "axes.labelcolor": PLOT_COLORS["text"],
                "text.color": PLOT_COLORS["text"],
                "xtick.color": PLOT_COLORS["text"],
                "ytick.color": PLOT_COLORS["text"],
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "grid.color": PLOT_COLORS["grid"],
                "grid.linestyle": "--",
                "grid.alpha": 0.45,
                "legend.frameon": True,
                "legend.facecolor": "#F7F4EF",
                "legend.edgecolor": "#BDB4AA",
            }
        )


class ResidualPlotStyle:
    @staticmethod
    def apply() -> None:
        plt.style.use("ggplot")
        plt.rcParams["figure.figsize"] = (10, 6)


class SourceDataAnalyzer:
    def __init__(self, project_root: Path, output_dir: Path) -> None:
        self.project_root = project_root
        self.output_dir = output_dir
        self.locator = DataLocator(project_root)

    def run(self) -> SourceArtifacts:
        SourcePlotStyle.apply()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        raw_frame, source_path = self._load_frame()
        columns = self._resolve_columns(raw_frame)
        clean_frame, removed_rows = self._clean_frame(raw_frame, columns)

        statistics_map = {
            "consumption": self._compute_statistics(clean_frame[columns.consumption]),
            "temperature": self._compute_statistics(clean_frame[columns.temperature]),
        }

        statistics_path = self._save_statistics(statistics_map)
        overview_path = self._save_overview(
            source_path=source_path,
            columns=columns,
            raw_rows=len(raw_frame),
            clean_rows=len(clean_frame),
            removed_rows=removed_rows,
            clean_frame=clean_frame,
        )

        figures = (
            self._save_distribution_plot(
                data=clean_frame[columns.consumption],
                statistics=statistics_map["consumption"],
                title="Распределение потребления электроэнергии",
                x_label="Потребление (МВтч)",
                output_path=self.output_dir / "consumption_dist.png",
                color=PLOT_COLORS["consumption"],
            ),
            self._save_distribution_plot(
                data=clean_frame[columns.temperature],
                statistics=statistics_map["temperature"],
                title="Распределение температуры",
                x_label="Температура (°C)",
                output_path=self.output_dir / "temperature_dist.png",
                color=PLOT_COLORS["temperature"],
            ),
            self._save_time_series_plot(
                frame=clean_frame,
                columns=columns,
                output_path=self.output_dir / "time_series.png",
            ),
        )

        return SourceArtifacts(
            source_path=source_path,
            output_dir=self.output_dir,
            statistics_path=statistics_path,
            overview_path=overview_path,
            figures=figures,
        )

    def _load_frame(self) -> tuple[pd.DataFrame, Path]:
        source_path = self.locator.find_source_data_path()
        frame = read_tabular_frame(source_path)
        return frame, source_path

    @staticmethod
    def _resolve_columns(frame: pd.DataFrame) -> DataColumns:
        time_column = SourceDataAnalyzer._find_column(frame, ("начало часа", "дата", "время"))
        consumption_column = SourceDataAnalyzer._find_column(frame, ("потребление", "мвтч"))
        temperature_column = SourceDataAnalyzer._find_column(frame, ("температура", "градус"))

        if not all((time_column, consumption_column, temperature_column)):
            raise DataAnalysisError("required source columns were not found")

        return DataColumns(
            time=time_column,
            consumption=consumption_column,
            temperature=temperature_column,
        )

    @staticmethod
    def _find_column(frame: pd.DataFrame, keywords: Iterable[str]) -> str | None:
        for column in frame.columns:
            lowered_name = str(column).lower()
            if any(keyword in lowered_name for keyword in keywords):
                return column
        return None

    @staticmethod
    def _clean_frame(frame: pd.DataFrame, columns: DataColumns) -> tuple[pd.DataFrame, int]:
        clean_frame = frame.copy()

        clean_frame[columns.consumption] = pd.to_numeric(
            clean_frame[columns.consumption]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.replace(",", "."),
            errors="coerce",
        )
        clean_frame[columns.temperature] = pd.to_numeric(
            clean_frame[columns.temperature].astype(str).str.replace(",", "."),
            errors="coerce",
        )
        clean_frame["datetime"] = SourceDataAnalyzer._parse_datetime_series(clean_frame[columns.time])

        raw_row_count = len(clean_frame)
        clean_frame = clean_frame.dropna(
            subset=["datetime", columns.consumption, columns.temperature]
        )
        clean_frame = clean_frame[
            np.isfinite(clean_frame[columns.consumption])
            & np.isfinite(clean_frame[columns.temperature])
        ].copy()

        if clean_frame.empty:
            raise DataAnalysisError("source dataset is empty after cleaning")

        removed_rows = raw_row_count - len(clean_frame)
        return clean_frame, removed_rows

    @staticmethod
    def _parse_datetime_series(series: pd.Series) -> pd.Series:
        parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
        if not parsed.isna().any():
            return parsed

        formats = (
            "%d.%m.%Y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%d.%m.%Y %H:%M:%S",
            "%Y.%m.%d %H:%M",
        )
        for fmt in formats:
            missing_mask = parsed.isna()
            if not missing_mask.any():
                break
            parsed.loc[missing_mask] = pd.to_datetime(
                series.loc[missing_mask],
                format=fmt,
                errors="coerce",
            )
        return parsed

    @staticmethod
    def _compute_statistics(series: pd.Series) -> SeriesStatistics:
        quantiles = series.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        mode_series = series.mode()
        mode_value = float(mode_series.iloc[0]) if not mode_series.empty else float("nan")

        return SeriesStatistics(
            count=int(series.count()),
            minimum=float(series.min()),
            maximum=float(series.max()),
            mean=float(series.mean()),
            std=float(series.std()),
            median=float(series.median()),
            mode=mode_value,
            skewness=float(series.skew()),
            kurtosis=float(series.kurtosis()),
            q01=float(quantiles.loc[0.01]),
            q05=float(quantiles.loc[0.05]),
            q25=float(quantiles.loc[0.25]),
            q50=float(quantiles.loc[0.5]),
            q75=float(quantiles.loc[0.75]),
            q95=float(quantiles.loc[0.95]),
            q99=float(quantiles.loc[0.99]),
        )

    def _save_statistics(self, statistics_map: dict[str, SeriesStatistics]) -> Path:
        statistics_path = self.output_dir / "source_statistics.csv"
        pd.DataFrame(
            [statistics.to_record(series_name) for series_name, statistics in statistics_map.items()]
        ).to_csv(statistics_path, index=False, encoding="utf-8-sig")
        return statistics_path

    def _save_overview(
        self,
        source_path: Path,
        columns: DataColumns,
        raw_rows: int,
        clean_rows: int,
        removed_rows: int,
        clean_frame: pd.DataFrame,
    ) -> Path:
        overview_path = self.output_dir / "source_overview.csv"
        overview_frame = pd.DataFrame(
            [
                {
                    "source_path": str(source_path),
                    "time_column": columns.time,
                    "consumption_column": columns.consumption,
                    "temperature_column": columns.temperature,
                    "raw_rows": raw_rows,
                    "clean_rows": clean_rows,
                    "removed_rows": removed_rows,
                    "time_start": clean_frame["datetime"].min(),
                    "time_end": clean_frame["datetime"].max(),
                }
            ]
        )
        overview_frame.to_csv(overview_path, index=False, encoding="utf-8-sig")
        return overview_path

    def _save_distribution_plot(
        self,
        data: pd.Series,
        statistics: SeriesStatistics,
        title: str,
        x_label: str,
        output_path: Path,
        color: str,
    ) -> Path:
        figure, axis = plt.subplots(figsize=(14, 9))

        sns.histplot(
            data,
            kde=True,
            bins=min(80, int(len(data) ** 0.5)),
            color=color,
            edgecolor="#F7F4EF",
            alpha=0.75,
            stat="density",
            ax=axis,
            line_kws={"linewidth": 2.8},
        )

        markers = (
            ("Среднее", statistics.mean, PLOT_COLORS["mean"], 2.3),
            ("Медиана", statistics.median, PLOT_COLORS["median"], 2.3),
            ("Мода", statistics.mode, PLOT_COLORS["mode"], 2.0),
        )
        for label, value, marker_color, line_width in markers:
            axis.axvline(
                value,
                color=marker_color,
                linestyle="--",
                linewidth=line_width,
                label=f"{label}: {value:.2f}",
            )

        annotation = "\n".join(
            (
                f"Асимметрия: {statistics.skewness:.4f}",
                f"Эксцесс: {statistics.kurtosis:.4f}",
                f"Стд. откл.: {statistics.std:.2f}",
                f"Объем выборки: {statistics.count:d}",
            )
        )
        axis.text(
            0.97,
            0.97,
            annotation,
            transform=axis.transAxes,
            fontsize=13,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": PLOT_COLORS["annotation_box"],
                "alpha": 0.92,
                "edgecolor": "#B8A999",
            },
            linespacing=1.5,
        )

        axis.set_title(title, pad=18)
        axis.set_xlabel(x_label, labelpad=10)
        axis.set_ylabel("Плотность распределения", labelpad=10)
        axis.grid(True, which="major")
        axis.minorticks_on()
        axis.grid(which="minor", linestyle=":", alpha=0.22)
        axis.legend(loc="upper left", fontsize=12, title="Статистики", title_fontsize=13)

        q_low = statistics.q01
        q_high = statistics.q99
        x_padding = (q_high - q_low) * 0.1
        axis.set_xlim(q_low - x_padding, q_high + x_padding)

        sns.despine(trim=True, offset=8)
        figure.tight_layout()
        figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=figure.get_facecolor())
        plt.close(figure)
        return output_path

    def _save_time_series_plot(
        self,
        frame: pd.DataFrame,
        columns: DataColumns,
        output_path: Path,
    ) -> Path:
        if "datetime" not in frame.columns or frame["datetime"].isna().all():
            raise DataAnalysisError("time series plot requires a valid datetime column")

        plot_frame = frame.sort_values("datetime").copy()
        if len(plot_frame) > 12000:
            step = max(len(plot_frame) // 12000, 1)
            plot_frame = plot_frame.iloc[::step].copy()

        figure, left_axis = plt.subplots(figsize=(16, 9))
        right_axis = left_axis.twinx()

        left_axis.set_xlabel("Дата")
        left_axis.set_ylabel("Потребление (МВтч)", color=PLOT_COLORS["consumption"])
        left_axis.plot(
            plot_frame["datetime"],
            plot_frame[columns.consumption],
            color=PLOT_COLORS["consumption"],
            linewidth=1.4,
            alpha=0.95,
            label="Потребление",
            zorder=3,
        )
        left_axis.fill_between(
            plot_frame["datetime"],
            plot_frame[columns.consumption],
            plot_frame[columns.consumption].min(),
            color=PLOT_COLORS["consumption"],
            alpha=0.10,
            zorder=2,
        )
        left_axis.tick_params(axis="y", labelcolor=PLOT_COLORS["consumption"])

        right_axis.set_ylabel("Температура (°C)", color=PLOT_COLORS["temperature"])
        right_axis.plot(
            plot_frame["datetime"],
            plot_frame[columns.temperature],
            color=PLOT_COLORS["temperature"],
            linewidth=1.2,
            alpha=0.88,
            label="Температура",
            zorder=4,
        )
        right_axis.tick_params(axis="y", labelcolor=PLOT_COLORS["temperature"])

        left_axis.xaxis.set_major_locator(mdates.YearLocator())
        left_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        left_axis.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
        figure.autofmt_xdate()

        left_axis.set_title(
            "Динамика потребления электроэнергии и температуры\n"
            f"Период: {plot_frame['datetime'].min().strftime('%d.%m.%Y')} - "
            f"{plot_frame['datetime'].max().strftime('%d.%m.%Y')}",
            pad=12,
        )
        left_axis.grid(True, which="major")
        left_axis.grid(True, which="minor", linestyle=":", alpha=0.18)

        lines_left, labels_left = left_axis.get_legend_handles_labels()
        lines_right, labels_right = right_axis.get_legend_handles_labels()
        legend = left_axis.legend(
            lines_left + lines_right,
            labels_left + labels_right,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            fontsize=12,
            title="Параметры",
            title_fontsize=13,
        )
        legend.get_frame().set_alpha(0.95)

        for spine in left_axis.spines.values():
            spine.set_alpha(0.35)
        for spine in right_axis.spines.values():
            spine.set_alpha(0.0)

        figure.tight_layout(rect=[0, 0.035, 1, 0.95])
        figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=figure.get_facecolor())
        plt.close(figure)
        return output_path


class ResidualDiagnosticsAnalyzer:
    def __init__(self, project_root: Path, output_dir: Path) -> None:
        self.project_root = project_root
        self.output_dir = output_dir
        self.locator = DataLocator(project_root)

    def run(self) -> ResidualArtifacts:
        ResidualPlotStyle.apply()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        scipy_stats, statsmodels_api, ljung_box = self._load_dependencies()
        residual_frame, input_path = self._load_residual_frame()

        summary = self._build_summary_record(residual_frame, scipy_stats)
        summary_path = self._save_summary(summary)
        ljung_box_path = self._save_ljung_box(residual_frame["residuals"], ljung_box)

        figures = (
            self._save_residual_histogram(
                residuals=residual_frame["residuals"],
                mean_value=float(summary["bias"]),
                std_value=float(summary["std"]),
                output_path=self.output_dir / "9_residuals_hist.png",
            ),
            self._save_acf_plot(
                residuals=residual_frame["residuals"],
                statsmodels_api=statsmodels_api,
                output_path=self.output_dir / "10a_residuals_acf.png",
            ),
            self._save_qq_plot(
                residuals=residual_frame["residuals"],
                scipy_stats=scipy_stats,
                output_path=self.output_dir / "10b_qq_plot.png",
            ),
            self._save_residual_series_plot(
                frame=residual_frame,
                output_path=self.output_dir / "10c_residuals_time.png",
            ),
        )

        return ResidualArtifacts(
            input_path=input_path,
            output_dir=self.output_dir,
            summary_path=summary_path,
            ljung_box_path=ljung_box_path,
            figures=figures,
        )

    @staticmethod
    def _load_dependencies():
        try:
            import statsmodels.api as statsmodels_api
            from scipy import stats as scipy_stats
            from statsmodels.stats.diagnostic import acorr_ljungbox
        except ImportError as exc:
            raise DependencyError("residual diagnostics dependencies are unavailable") from exc
        return scipy_stats, statsmodels_api, acorr_ljungbox

    def _load_residual_frame(self) -> tuple[pd.DataFrame, Path]:
        input_path = self.locator.find_catboost_predictions_path()
        frame = read_tabular_frame(input_path)

        required_columns = {"y", "y_hat"}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise DataAnalysisError(f"prediction file is missing required columns: {missing}")

        residual_frame = frame.copy()
        residual_frame["residuals"] = residual_frame["y"] - residual_frame["y_hat"]
        residual_frame = residual_frame.loc[residual_frame["residuals"].notna()].copy()

        if residual_frame.empty:
            raise DataAnalysisError("residual series is empty")

        if "start_dt" in residual_frame.columns:
            residual_frame["start_dt"] = pd.to_datetime(
                residual_frame["start_dt"],
                errors="coerce",
            )

        return residual_frame, input_path

    @staticmethod
    def _build_summary_record(frame: pd.DataFrame, scipy_stats) -> dict[str, float]:
        residuals = frame["residuals"]
        summary = {
            "count": int(residuals.count()),
            "bias": float(residuals.mean()),
            "std": float(residuals.std()),
            "skewness": float(scipy_stats.skew(residuals)),
            "kurtosis": float(scipy_stats.kurtosis(residuals)),
            "mae": float(np.mean(np.abs(residuals.to_numpy(dtype=float)))),
        }

        try:
            shapiro_stat, shapiro_p = scipy_stats.shapiro(residuals)
            summary["shapiro_stat"] = float(shapiro_stat)
            summary["shapiro_p"] = float(shapiro_p)
        except Exception:
            summary["shapiro_stat"] = float("nan")
            summary["shapiro_p"] = float("nan")

        return summary

    def _save_summary(self, summary: dict[str, float]) -> Path:
        summary_path = self.output_dir / "residual_diagnostics_summary.csv"
        pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")
        return summary_path

    def _save_ljung_box(self, residuals: pd.Series, ljung_box) -> Path:
        ljung_box_path = self.output_dir / "residual_ljung_box.csv"
        ljung_box_frame = ljung_box(residuals, lags=[24, 48], return_df=True)
        ljung_box_frame.to_csv(ljung_box_path, encoding="utf-8-sig")
        return ljung_box_path

    def _save_residual_histogram(
        self,
        residuals: pd.Series,
        mean_value: float,
        std_value: float,
        output_path: Path,
    ) -> Path:
        figure, axis = plt.subplots()
        sns.histplot(residuals, kde=True, color="green", bins=30, ax=axis)
        axis.set_title(f"Распределение остатков (Mean={mean_value:.2f}, Std={std_value:.2f})")
        axis.set_xlabel("Ошибка (МВтч)")
        axis.set_ylabel("Частота")
        axis.axvline(x=mean_value, color="red", linestyle="--", label="Среднее")
        axis.legend()
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def _save_acf_plot(self, residuals: pd.Series, statsmodels_api, output_path: Path) -> Path:
        figure, axis = plt.subplots()
        statsmodels_api.graphics.tsa.plot_acf(
            residuals,
            lags=48,
            ax=axis,
            title="Автокорреляция остатков (ACF)",
        )
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def _save_qq_plot(self, residuals: pd.Series, scipy_stats, output_path: Path) -> Path:
        figure, axis = plt.subplots()
        scipy_stats.probplot(residuals, dist="norm", plot=axis)
        axis.set_title("Q-Q Plot")
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def _save_residual_series_plot(self, frame: pd.DataFrame, output_path: Path) -> Path:
        figure, axis = plt.subplots(figsize=(12, 5))

        x_axis: pd.Series | np.ndarray
        if "start_dt" in frame.columns and frame["start_dt"].notna().any():
            x_axis = frame["start_dt"]
        else:
            x_axis = frame.index.to_numpy()

        axis.plot(x_axis, frame["residuals"], color="purple", linewidth=1)
        axis.axhline(0, color="black", linestyle="--")
        axis.set_title("Динамика остатков во времени")
        axis.set_xlabel("Время")
        axis.set_ylabel("Ошибка (МВтч)")
        figure.tight_layout()
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        return output_path


def analyze_source_data(output_dir: Path = SOURCE_OUTPUT_DIR) -> bool:
    analyzer = SourceDataAnalyzer(PROJECT_ROOT, Path(output_dir))
    analyzer.run()
    return True


def analyze_residuals(output_dir: Path = RESIDUALS_OUTPUT_DIR) -> bool:
    analyzer = ResidualDiagnosticsAnalyzer(PROJECT_ROOT, Path(output_dir))
    analyzer.run()
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("all", "source", "residuals"),
        default="all",
        metavar="mode",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    try:
        if arguments.mode in {"all", "source"}:
            analyze_source_data()
        if arguments.mode in {"all", "residuals"}:
            analyze_residuals()
    except DataAnalysisError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
