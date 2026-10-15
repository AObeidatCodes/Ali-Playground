"""Peak picking and band assignment against a band table."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

BAND_TABLE = Path(__file__).resolve().parent.parent / "data" / "band_table.csv"


def load_band_table(path=BAND_TABLE):
    return pd.read_csv(path)


def pick_peaks(wavenumber, corrected, prominence=0.08, max_peaks=12):
    """Strongest peaks of a 0-1 scaled, baseline-corrected spectrum."""
    idx, props = find_peaks(np.nan_to_num(corrected), prominence=prominence)
    order = np.argsort(props["prominences"])[::-1][:max_peaks]
    peaks = pd.DataFrame({
        "wavenumber": wavenumber[idx[order]].round(0),
        "height": corrected[idx[order]].round(2),
    })
    return peaks.sort_values("wavenumber", ascending=False).reset_index(drop=True)


def assign(peaks, table=None):
    """Add every band-table row whose range contains the peak."""
    table = load_band_table() if table is None else table
    labels = []
    for wn in peaks["wavenumber"]:
        rows = table[(table["low"] <= wn) & (table["high"] >= wn)]
        labels.append("; ".join(rows["assignment"]) if len(rows) else "")
    return peaks.assign(assignment=labels)
