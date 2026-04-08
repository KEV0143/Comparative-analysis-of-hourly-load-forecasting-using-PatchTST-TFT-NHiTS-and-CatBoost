from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_tabular_frame(source_path: Path) -> pd.DataFrame:
    if source_path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(source_path, engine="openpyxl")
    else:
        frame = pd.read_csv(source_path, sep=None, engine="python")
    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    return frame
