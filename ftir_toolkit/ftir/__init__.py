"""Small FTIR toolkit: read spectra, correct baselines, assign bands and
search a reference library by correlation."""
from .io import Spectrum, read_folder, read_spectrum
from .match import Library, verdict
