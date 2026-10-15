"""Build the reference library and the blind exercise spectra.

Do not read the EXERCISES mapping below before you have done the exercises:
it gives the answers away.

Sources (all real measurements):
  - Open Specy package (CC BY 4.0): https://github.com/wincowgerDEV/OpenSpecy-package
      data/test_lib.RData   -> 29 FTIR reference spectra (library)
      inst/extdata/*        -> measured unknowns (set B)
  - NIST gas-phase IR spectra, as redistributed in the jcamp package
    (https://github.com/nzhagen/jcamp/tree/master/data/infrared_spectra).
    NIST data is copyrighted, so set A is downloaded here, not committed.

Usage:
  git clone --depth 1 https://github.com/wincowgerDEV/OpenSpecy-package
  pip install rdata xarray jcamp
  python build_exercises.py --openspecy OpenSpecy-package
"""
import argparse
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ftir.io import read_spectrum

HERE = Path(__file__).resolve().parent
LIB_DIR = HERE / "data" / "library"
SET_A_DIR = HERE / "data" / "unknowns_nist"   # git-ignored
SET_B_DIR = HERE / "data" / "unknowns"
NIST_RAW = "https://raw.githubusercontent.com/nzhagen/jcamp/master/data/infrared_spectra/"

# Readable names for the library's SpectrumIdentity values.
MATERIAL = {
    "PS": "polystyrene", "PET": "polyethylene terephthalate (PET)", "Soil": "soil",
    "LDPE": "low-density polyethylene (LDPE)", "polycarbonate": "polycarbonate",
    "ABS": "ABS", "PVC.1": "polyvinyl chloride (PVC)", "PA66": "polyamide 6.6 (nylon)",
    "PEST": "polyester", "CA": "cellulose acetate", "polyacetal": "polyacetal (POM)",
    "TPU": "thermoplastic polyurethane", "PA": "polyacrylamide",
    "fibre_tussah_silk": "silk (protein fibre)", "phenoxy_resin": "phenoxy / epoxy resin",
    "polymethyl methacrylate": "PMMA", "poly(vinyl_alcohol)": "polyvinyl alcohol (PVA)",
    "poly(vinyl_acetate)": "polyvinyl acetate", "1_2_polybutadiene": "1,2-polybutadiene",
    "polyvinylpyrrolidone": "polyvinylpyrrolidone", "Nitrile": "nitrile rubber",
    "polyphenylsulfone": "polyphenylsulfone", "fibre_polyetheretherketone": "PEEK",
    "PDMS": "silicone (PDMS)", "poly(2 hydroxyethyl methacrylate)": "poly(HEMA)",
    "poly(2 6 dimethyl p phenylene oxide)": "polyphenylene oxide (PPO)",
    "poly(acrylic acid)": "polyacrylic acid",
    "methyl vinyl ether maleic acid": "methyl vinyl ether / maleic acid copolymer",
    "Barrerite__R050135-1__Infrared__Infrared_Data_Processed__767": "barrerite (zeolite mineral, RRUFF)",
}

# Exercise id -> source. Set A: NIST gas phase. Set B: measured samples.
EXERCISES = {
    "A1": ("nist", "acetone.jdx"),
    "A2": ("nist", "water.jdx"),
    "A3": ("nist", "sulphur hexafluoride.jdx"),
    "A4": ("nist", "isopropyl alcohol.jdx"),
    "A5": ("nist", "carbon dioxide.jdx"),
    "B1": ("map", (6, 5)),          # centre pixel of the cellulose acetate map
    "B2": ("extdata", "ftir_ldpe_soil.asp"),
    "B3": ("extdata", "fitr_nitrocellulose.jdx"),
    "B4": ("map", (0, 15)),         # corner pixel of the same map
    "B5": ("extdata", "ftir_pva_without_header.csv"),
}


def build_library(openspecy):
    import rdata

    lib = rdata.read_rda(str(openspecy / "data" / "test_lib.RData"))["test_lib"]
    grid = np.asarray(lib["wavenumber"], float)
    spectra = np.asarray(lib["spectra"], float)
    meta = lib["metadata"]
    keep = [i for i, kind in enumerate(meta["SpectrumType"]) if kind == "FTIR"]

    ids = [f"OS{i:02d}" for i in keep]
    wide = pd.DataFrame(spectra[:, keep], columns=ids)
    wide.insert(0, "wavenumber", grid)
    rows = []
    for i, sid in zip(keep, ids):
        identity = str(meta["SpectrumIdentity"].iloc[i])
        rows.append({
            "id": sid,
            "material": MATERIAL.get(identity, identity),
            "library_label": identity,
            "mode": meta["SpectralCollectionMode"].iloc[i],
            "contributor": meta["Organization"].iloc[i],
            "license": "CC BY 4.0, Open Specy test_lib",
        })
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_csv(LIB_DIR / "openspecy_ftir.csv", index=False, float_format="%.5f")
    pd.DataFrame(rows).to_csv(LIB_DIR / "openspecy_ftir_metadata.csv", index=False)
    print(f"library: {len(ids)} FTIR spectra")


def read_envi_pixel(zip_path, line, sample):
    with zipfile.ZipFile(zip_path) as zf:
        hdr = zf.read("CA small UF.hdr").decode()
        cube = np.frombuffer(zf.read("CA small UF.dat"), dtype="<f4")
    dims = {k: int(re.search(rf"{k} = (\d+)", hdr).group(1)) for k in ("samples", "lines", "bands")}
    wn = np.array([float(v) for v in re.search(r"wavelength = \{([^}]*)\}", hdr).group(1).split(",")])
    cube = cube.reshape(dims["lines"], dims["samples"], dims["bands"])  # band-interleaved by pixel
    return wn, cube[line, sample, :]


def write_csv(path, wn, absorbance):
    order = np.argsort(wn)
    with open(path, "w") as f:
        f.write("# unit: absorbance\nwavenumber,absorbance\n")
        for x, y in zip(np.asarray(wn)[order], np.asarray(absorbance)[order]):
            f.write(f"{x:.2f},{y:.6g}\n")


def build_unknowns(openspecy):
    extdata = openspecy / "inst" / "extdata"
    SET_A_DIR.mkdir(parents=True, exist_ok=True)
    SET_B_DIR.mkdir(parents=True, exist_ok=True)
    tmp = HERE / "data" / ".download"
    tmp.mkdir(exist_ok=True)
    for ex_id, (kind, source) in EXERCISES.items():
        if kind == "nist":
            local = tmp / source
            if not local.exists():
                urllib.request.urlretrieve(NIST_RAW + urllib.parse.quote(source), local)
            s = read_spectrum(local)
            write_csv(SET_A_DIR / f"{ex_id}.csv", s.wavenumber, s.absorbance)
        elif kind == "map":
            wn, y = read_envi_pixel(extdata / "CA_tiny_map.zip", *source)
            write_csv(SET_B_DIR / f"{ex_id}.csv", wn, y)
        else:
            s = read_spectrum(extdata / source)
            write_csv(SET_B_DIR / f"{ex_id}.csv", s.wavenumber, s.absorbance)
        print(f"{ex_id} written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--openspecy", type=Path, required=True,
                        help="path to a clone of the OpenSpecy-package repository")
    parser.add_argument("--skip-library", action="store_true")
    args = parser.parse_args()
    if not args.skip_library:
        build_library(args.openspecy)
    build_unknowns(args.openspecy)
