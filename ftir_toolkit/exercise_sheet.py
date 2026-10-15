"""One printable sheet per exercise set: every spectrum baseline-corrected,
stacked, with its strongest peaks labelled and no library answer shown.

Example:
  python exercise_sheet.py data/unknowns --out sheet_B.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ftir import read_folder
from ftir.bands import pick_peaks
from ftir.process import corrected_absorbance


def main():
    parser = argparse.ArgumentParser(description="Blind exercise sheet")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=Path("exercise_sheet.png"))
    args = parser.parse_args()

    spectra = read_folder(args.folder)
    fig, axes = plt.subplots(len(spectra), 1, figsize=(10, 2.3 * len(spectra)), sharex=True)
    for ax, s in zip(axes, spectra):
        corrected = corrected_absorbance(s.wavenumber, s.absorbance)
        ax.plot(s.wavenumber, corrected, color="black", lw=0.9)
        for _, p in pick_peaks(s.wavenumber, corrected, max_peaks=8).iterrows():
            ax.annotate(f"{p.wavenumber:.0f}", (p.wavenumber, p.height), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7)
        ax.set_ylim(-0.05, 1.2)
        ax.set_yticks([])
        ax.text(0.01, 0.82, s.name, transform=ax.transAxes, fontsize=11, weight="bold")
    axes[-1].set_xlim(4000, 600)
    axes[-1].set_xlabel("wavenumber (cm$^{-1}$)")
    fig.supylabel("absorbance, baseline-corrected")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"sheet -> {args.out}")


if __name__ == "__main__":
    main()
