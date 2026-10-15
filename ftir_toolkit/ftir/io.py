"""Read FTIR spectra from CSV, Agilent .asp and JCAMP-DX files.

Every reader returns a Spectrum with wavenumbers in cm-1 (ascending) and
intensities as absorbance. Transmittance and reflectance are converted on
the way in, so the rest of the pipeline only ever sees absorbance.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SUPPORTED = {".csv", ".txt", ".asp", ".jdx", ".dx", ".jcamp"}


@dataclass
class Spectrum:
    name: str
    wavenumber: np.ndarray
    absorbance: np.ndarray
    meta: dict = field(default_factory=dict)


def read_spectrum(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        wn, y, meta = _read_csv(path)
    elif suffix == ".asp":
        wn, y, meta = _read_asp(path)
    elif suffix in (".jdx", ".dx", ".jcamp"):
        wn, y, meta = _read_jcamp(path)
    else:
        raise ValueError(f"Unsupported file type: {path.name}")

    wn, y = np.asarray(wn, float), np.asarray(y, float)
    keep = np.isfinite(wn) & np.isfinite(y)
    wn, y = wn[keep], y[keep]
    order = np.argsort(wn)
    return Spectrum(path.stem, wn[order], y[order], meta)


def read_folder(folder):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in SUPPORTED)
    return [read_spectrum(p) for p in files]


def _read_csv(path):
    # Two numeric columns (wavenumber, intensity); a header row and lines
    # starting with '#' are allowed. A '# unit: transmittance' line converts.
    meta, unit = {}, "absorbance"
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") and ":" in line:
                key, value = line[1:].split(":", 1)
                meta[key.strip().lower()] = value.strip()
    unit = meta.get("unit", unit).lower()
    df = pd.read_csv(path, comment="#", header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    wn, y = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    return wn, _to_absorbance(y, unit), meta


def _read_asp(path):
    # Agilent .asp: n points, first x, last x, three instrument values, then n intensities.
    values = [float(v) for v in Path(path).read_text().split()]
    n, first_x, last_x = int(values[0]), values[1], values[2]
    y = np.array(values[6:6 + n])
    wn = np.linspace(first_x, last_x, n)
    return wn, y, {"format": "Agilent .asp"}


def _read_jcamp(path):
    import jcamp  # pip install jcamp

    d = jcamp.readfile(str(path))
    wn, y = np.asarray(d["x"], float), np.asarray(d["y"], float)
    xunits = str(d.get("xunits", "1/cm")).lower()
    if "micro" in xunits or xunits.strip() in ("um", "µm"):
        wn = 1e4 / wn
    meta = {k: d[k] for k in ("title", "origin", "owner", "state", "yunits") if k in d}
    return wn, _to_absorbance(y, str(d.get("yunits", "absorbance"))), meta


def _to_absorbance(y, unit):
    unit = unit.lower()
    if "transmit" in unit or "reflect" in unit:
        if np.nanmax(y) > 1.5:  # percent
            y = y / 100.0
        return -np.log10(np.clip(y, 1e-6, None))
    return y
