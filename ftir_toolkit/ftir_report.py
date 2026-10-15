"""Batch FTIR report: baseline, band assignment and library search for every
spectrum in a folder, with one figure per spectrum and a summary table.

Example:
  python ftir_report.py data/unknowns --out reports
  python ftir_report.py data/unknowns_nist --out reports_nist --blind
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ftir import Library, read_folder, verdict
from ftir.bands import assign, pick_peaks
from ftir.process import corrected_absorbance

HERE = Path(__file__).resolve().parent
DEFAULT_LIBRARY = HERE / "data" / "library" / "openspecy_ftir.csv"


def analyse(spectrum, library):
    corrected = corrected_absorbance(spectrum.wavenumber, spectrum.absorbance)
    peaks = assign(pick_peaks(spectrum.wavenumber, corrected))
    hits, query = library.search(spectrum)
    return corrected, peaks, hits, query


def plot(spectrum, corrected, peaks, hits, query, library, path, blind=False):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2]})
    top.plot(spectrum.wavenumber, corrected, color="black", lw=1)
    for _, p in peaks.iterrows():
        top.annotate(f"{p.wavenumber:.0f}", (p.wavenumber, p.height),
                     textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    top.set_ylabel("absorbance, baseline-corrected (0-1)")
    top.set_title(f"{spectrum.name}" + ("" if blind else f" - best hit: {hits.iloc[0].material} (r = {hits.iloc[0].r:.2f})"))

    bottom.plot(library.grid, query, color="black", lw=1, label="unknown")
    if not blind:
        best = library.matrix[:, library.ids.index(hits.iloc[0].id)]
        bottom.plot(library.grid, best, color="tab:blue", lw=1, alpha=0.7, label=f"library: {hits.iloc[0].material}")
        bottom.legend(loc="upper left", fontsize=8)
    bottom.set_ylabel("|1st derivative| (0-1)")
    bottom.set_xlabel("wavenumber (cm$^{-1}$)")
    bottom.set_xlim(4000, 600)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Batch FTIR report")
    parser.add_argument("folder", type=Path, help="folder with .csv / .asp / .jdx spectra")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--blind", action="store_true",
                        help="figures without library hits, for practising band assignment")
    args = parser.parse_args()

    start = time.perf_counter()
    library = Library(args.library)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, sections = [], []
    for spectrum in read_folder(args.folder):
        corrected, peaks, hits, query = analyse(spectrum, library)
        plot(spectrum, corrected, peaks, hits, query, library,
             args.out / f"{spectrum.name}.png", blind=args.blind)
        call = verdict(hits)
        rows.append({"spectrum": spectrum.name, "best_hit": hits.iloc[0].material,
                     "r": round(hits.iloc[0].r, 3), "second_hit": hits.iloc[1].material,
                     "r_second": round(hits.iloc[1].r, 3), "verdict": call})
        sections.append(_section(spectrum.name, peaks, hits, call, args.blind))

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "summary.csv", index=False)
    elapsed = time.perf_counter() - start
    header = (f"# FTIR report: {args.folder}\n\n{len(rows)} spectra, library "
              f"{args.library.name} ({len(library.ids)} spectra), {elapsed:.1f} s.\n\n")
    table = "" if args.blind else summary.to_markdown(index=False) + "\n\n"
    (args.out / "report.md").write_text(header + table + "\n".join(sections))
    print(f"{len(rows)} spectra in {elapsed:.1f} s -> {args.out}/report.md")


def _section(name, peaks, hits, call, blind):
    text = f"## {name}\n\n![{name}]({name}.png)\n\n**Peaks and possible assignments**\n\n"
    text += peaks.to_markdown(index=False) + "\n\n"
    if not blind:
        cols = [c for c in ("material", "r", "mode", "contributor") if c in hits]
        text += f"**Library hits** - verdict: *{call}*\n\n" + hits[cols].round(3).to_markdown(index=False) + "\n\n"
    return text


if __name__ == "__main__":
    main()
