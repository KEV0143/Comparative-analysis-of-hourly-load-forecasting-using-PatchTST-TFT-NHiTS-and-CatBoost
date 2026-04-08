from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


class ForecastFigureStyle:
    figure_size = (12.8, 6.45)
    dpi = 150
    actual_color = "#272829"
    model_color = "#0D53E0"
    grid_major = "#D5DBE6"
    grid_minor = "#ECEFF4"
    spine_color = "#8B919A"
    text_color = "#1E1E1E"

    @classmethod
    def apply(cls) -> None:
        plt.rcParams.update(
            {
                "font.family": "DejaVu Serif",
                "font.size": 11,
                "figure.facecolor": "#FFFFFF",
                "savefig.facecolor": "#FFFFFF",
                "axes.facecolor": "#FFFFFF",
                "axes.edgecolor": cls.spine_color,
                "axes.labelcolor": cls.text_color,
                "xtick.color": cls.text_color,
                "ytick.color": cls.text_color,
                "text.color": cls.text_color,
                "legend.frameon": False,
            }
        )

    @classmethod
    def create_figure(cls):
        return plt.subplots(figsize=cls.figure_size, dpi=cls.dpi)

    @classmethod
    def style_axis(cls, axis, title: str) -> None:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)
        axis.minorticks_on()
        axis.grid(True, which="major", color=cls.grid_major, linewidth=0.8, alpha=0.75)
        axis.grid(True, which="minor", color=cls.grid_minor, linewidth=0.6, alpha=0.45)
        axis.set_title(title, fontsize=15, pad=12)
        axis.set_xlabel("Дата и время")
        axis.set_ylabel("Потребление, МВтч")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(cls.spine_color)
        axis.spines["bottom"].set_color(cls.spine_color)
        axis.tick_params(axis="both", labelsize=11)
        axis.margins(x=0.01)
