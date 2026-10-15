# ftir_toolkit

Batch evaluation of FTIR spectra: read the files, correct the baseline, label the bands, search a
reference library, and write one report with a figure per spectrum. Built as preparation for a
contamination-analysis role, to do for a whole folder of spectra what is otherwise done by hand,
spectrum by spectrum.

```
python ftir_report.py data/unknowns --out reports
```

```
5 spectra in 1.6 s -> reports/report.md
```

## What it does

1. **Reads** CSV, Agilent `.asp` and JCAMP-DX (`.jdx`) files. Transmittance and reflectance are
   converted to absorbance on the way in (`ftir/io.py`).
2. **Corrects the baseline** with asymmetric least squares (Eilers & Boelens 2005) and scales each
   spectrum to 0-1 (`ftir/process.py`).
3. **Picks peaks and assigns them** against `data/band_table.csv`, which includes the CO2 and
   water-vapour artefacts (`ftir/bands.py`).
4. **Searches the library** the way Open Specy does: 5 cm-1 grid, Savitzky-Golay first derivative
   (window 11, polynomial 3), absolute value, min-max normalisation, then Pearson correlation. The
   CO2 region 2280-2400 cm-1 is masked (`ftir/match.py`).
5. **Gives a verdict**: *probable match* (r >= 0.70 and at least 0.05 ahead of the next
   material), *weak* (r >= 0.50), or *no reliable match*. These thresholds are rules of thumb.
   Check them with known samples.
6. **Writes** `report.md`, `summary.csv` and one PNG per spectrum (`ftir_report.py`).

A derivative-based search ignores sloped baselines without having to model them. The verdict looks
at the gap to the second-best material as well as the top score, because two hits close together
mean the search could not tell them apart.

## Results on the measured exercise spectra

| Spectrum | Best hit | r | Verdict | Correct? |
| --- | --- | --- | --- | --- |
| B1 | cellulose acetate | 0.91 | probable match | yes |
| B2 | polystyrene | 0.57 | weak | **no**: it is polyethylene. The band table and the figure show it. |
| B3 | polyacrylic acid | 0.49 | no reliable match | yes: nitrocellulose, not in the library |
| B4 | silicone (PDMS) | 0.27 | no reliable match | yes: empty pixel |
| B5 | barrerite (silicate) | 0.51 | weak | partly: a silicate-filled, acetate-containing material |

The tool is right four times out of five, and its weak verdict on B2 tells you to check that one.
That is the honest way to use a library search: as a hypothesis that the band assignment confirms
or rejects.

## Practice exercises

- `data/unknowns/` holds set B: five real measurements, committed.
- Set A (five NIST gas-phase spectra) is downloaded by `build_exercises.py` into
  `data/unknowns_nist/`, because NIST data is copyrighted and not redistributed here.
- `python exercise_sheet.py data/unknowns --out sheet_B.png` draws a blind sheet.
- Answers: `ANSWER_KEY.md`. Do not read it, or `build_exercises.py`, before you try.

## Setup

```
pip install -r requirements.txt
python -m pytest tests
```

Rebuilding the library and set A needs a clone of the Open Specy package and `pip install rdata xarray`:

```
git clone --depth 1 https://github.com/wincowgerDEV/OpenSpecy-package
python build_exercises.py --openspecy OpenSpecy-package
```

## Limits

- The library is Open Specy's 29-spectrum test library, not a production library. Real labs use
  commercial libraries with thousands of spectra and their own in-house references.
- Library and unknown should be measured the same way (ATR vs transmission vs reflectance). B2
  shows what happens when they are not.
- No ATR correction, spectral subtraction for mixtures, or map (hyperspectral) processing yet.
  Those are the next steps.

## Data and attribution

- Reference library and set B: Open Specy, CC BY 4.0. Cowger et al. (2021),
  *Anal. Chem.* 93, 7543-7548, and the Open Specy contributors listed in
  `data/library/openspecy_ftir_metadata.csv`.
  https://github.com/wincowgerDEV/OpenSpecy-package
- B3 (nitrocellulose): Pacific Northwest National Laboratory under IARPA contract, public domain.
- Set A: NIST gas-phase infrared spectra (Quant-IR, Coblentz Society, EPA collections), as
  distributed in the `jcamp` package (https://github.com/nzhagen/jcamp).
