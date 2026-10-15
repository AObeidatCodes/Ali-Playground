import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ftir import Library, Spectrum, read_folder, verdict  # noqa: E402
from ftir.bands import assign, pick_peaks  # noqa: E402
from ftir.process import abs_derivative, baseline_als, corrected_absorbance  # noqa: E402

LIBRARY = ROOT / "data" / "library" / "openspecy_ftir.csv"


def gaussian(x, centre, width, height=1.0):
    return height * np.exp(-0.5 * ((x - centre) / width) ** 2)


def test_baseline_removes_slope():
    x = np.linspace(600, 4000, 1500)
    band = gaussian(x, 1730, 15)
    corrected = corrected_absorbance(x, band + 0.0002 * x + 0.3)
    assert abs(x[np.argmax(corrected)] - 1730) < 5
    assert np.median(corrected) < 0.05


def test_derivative_ignores_offset():
    y = gaussian(np.linspace(600, 4000, 700), 2900, 20)
    assert np.allclose(abs_derivative(y), abs_derivative(y + 5.0), equal_nan=True)


def test_band_assignment_finds_carbonyl():
    x = np.linspace(600, 4000, 1500)
    peaks = assign(pick_peaks(x, gaussian(x, 1735, 12)))
    assert "C=O" in peaks.loc[0, "assignment"]


def test_transmittance_is_converted_to_absorbance():
    from ftir.io import _to_absorbance

    assert np.allclose(_to_absorbance(np.array([100.0, 10.0]), "% transmittance"), [0.0, 1.0])
    assert np.allclose(_to_absorbance(np.array([0.5]), "absorbance"), [0.5])


def test_measured_cellulose_acetate_is_found():
    lib = Library(LIBRARY)
    b1 = {s.name: s for s in read_folder(ROOT / "data" / "unknowns")}["B1"]
    hits, _ = lib.search(b1)
    assert hits.iloc[0].material == "cellulose acetate"
    assert verdict(hits).startswith("probable")


def test_flat_spectrum_is_no_match():
    x = np.linspace(600, 4000, 2000)
    noise = np.random.default_rng(0).normal(0, 0.001, x.size)
    hits, _ = Library(LIBRARY).search(Spectrum("blank", x, noise))
    assert verdict(hits).startswith("no reliable")
