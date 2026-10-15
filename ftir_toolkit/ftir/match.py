"""Library search: Pearson correlation between derivative spectra.

The correlation coefficient r is the hit quality index (HQI) here, as in
Open Specy. A top hit is a hypothesis, not an answer: the verdict below also
looks at how far the best hit is ahead of the second-best material.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .process import abs_derivative, co2_mask, to_grid

# Rule-of-thumb thresholds. Check them with known samples before you trust them.
PROBABLE_R = 0.70
WEAK_R = 0.50
MIN_MARGIN = 0.05


class Library:
    """Reference spectra stored as one wide CSV (wavenumber + one column per
    spectrum, already in derivative form) plus a metadata CSV keyed by id."""

    def __init__(self, spectra_csv, metadata_csv=None):
        wide = pd.read_csv(spectra_csv)
        self.grid = wide.iloc[:, 0].to_numpy(float)
        self.ids = list(wide.columns[1:])
        self.matrix = wide.iloc[:, 1:].to_numpy(float)
        if metadata_csv is None:
            metadata_csv = Path(spectra_csv).with_name(Path(spectra_csv).stem + "_metadata.csv")
        if Path(metadata_csv).exists():
            self.meta = pd.read_csv(metadata_csv).set_index("id")
        else:
            self.meta = pd.DataFrame(index=self.ids, data={"material": self.ids})

    def prepare(self, spectrum):
        """Put an unknown on the library grid in the same derivative form."""
        on_grid = to_grid(spectrum.wavenumber, spectrum.absorbance, self.grid)
        return abs_derivative(on_grid)

    def search(self, spectrum, top_n=5, mask_co2=True):
        query = self.prepare(spectrum)
        skip = co2_mask(self.grid) if mask_co2 else np.zeros_like(self.grid, bool)
        scores = [_pearson(query, self.matrix[:, j], skip) for j in range(len(self.ids))]
        hits = pd.DataFrame({"id": self.ids, "r": scores})
        hits = hits.join(self.meta, on="id").sort_values("r", ascending=False)
        hits = hits.drop_duplicates(subset="material") if "material" in hits else hits
        return hits.head(top_n).reset_index(drop=True), query


def verdict(hits):
    """Plain-language call on the top hit: probable match, weak, or none."""
    best = hits.iloc[0]
    margin = best.r - hits.iloc[1].r if len(hits) > 1 else 1.0
    if best.r >= PROBABLE_R and margin >= MIN_MARGIN:
        return "probable match - confirm 3-5 main bands"
    if best.r >= WEAK_R:
        return "weak - check bands, mixture or material not in library?"
    return "no reliable match - not in library, or escalate (SEM-EDX / TOF-SIMS)"


def _pearson(a, b, skip):
    ok = np.isfinite(a) & np.isfinite(b) & ~skip
    if ok.sum() < 20:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])
