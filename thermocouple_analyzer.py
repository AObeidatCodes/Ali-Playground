"""
Thermocouple Analysis & Visualization Tool
===========================================
A comprehensive tool for analyzing time-series data from thermocouple test benches.
Supports CSV/Excel data loading, drift analysis, and rich interactive visualizations
including raw temperature plots, drift plots, static contour maps, and animated
dynamic contour maps.

Usage
-----
1. Set RUN_TEST_MODE = False and run the script to open a file dialog for real data.
2. Set RUN_TEST_MODE = True and choose a TEST_SCENARIO to demo with synthetic data.
3. Adjust USER_CONFIG to match your thermocouple channel names and spatial layout.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional imports – raise a friendly error if not installed
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import plotly.io as pio
except ImportError as exc:
    sys.exit(
        "plotly is required.  Install with:  pip install plotly\n" + str(exc)
    )

try:
    from scipy.interpolate import griddata
except ImportError as exc:
    sys.exit(
        "scipy is required.  Install with:  pip install scipy\n" + str(exc)
    )

warnings.filterwarnings("ignore")


def _show_fig(fig: "go.Figure") -> None:
    """
    Attempt to display a Plotly figure.

    Silently skips display in headless / CI environments where no browser
    or IPython kernel is available.
    """
    try:
        fig.show()
    except Exception:  # noqa: BLE001
        pass


# ============================================================
#  USER CONFIGURATION
#  Edit this section to match your test-bench channel names
#  and physical layout.  No changes to the class code needed.
# ============================================================
USER_CONFIG: Dict = {
    # -----------------------------------------------------------
    # Thermocouple column names (as they appear in the data file)
    # -----------------------------------------------------------
    "thermocouple_columns": [
        "70418_T901000_W1T",  # Control / reference thermocouple
        "70418_T901000_W2T",
        "70418_T901000_W3T",
        "70418_T901000_W4T",
        "70418_T901000_W5T",
        "70418_T901000_W6T",
        "70418_T901000_W7T",
        "70418_T901000_W8T",
    ],
    # -----------------------------------------------------------
    # Human-readable legend labels (same order as above)
    # -----------------------------------------------------------
    "legend_labels": [
        "TC1 – Control (W1T)",
        "TC2 (W2T)",
        "TC3 (W3T)",
        "TC4 (W4T)",
        "TC5 (W5T)",
        "TC6 (W6T)",
        "TC7 (W7T)",
        "TC8 (W8T)",
    ],
    # -----------------------------------------------------------
    # The single control/reference thermocouple for drift analysis
    # -----------------------------------------------------------
    "control_tc": "70418_T901000_W1T",
    # -----------------------------------------------------------
    # Plot colours (Plotly colour strings or hex codes)
    # -----------------------------------------------------------
    "colors": [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # grey
    ],
    # -----------------------------------------------------------
    # Spatial map  {column_name: (x_mm, y_mm)}
    # Positions in any consistent unit (millimetres recommended)
    # -----------------------------------------------------------
    "spatial_map": {
        "70418_T901000_W1T": (0.0, 0.0),
        "70418_T901000_W2T": (25.0, 0.0),
        "70418_T901000_W3T": (50.0, 0.0),
        "70418_T901000_W4T": (75.0, 0.0),
        "70418_T901000_W5T": (0.0, 25.0),
        "70418_T901000_W6T": (25.0, 25.0),
        "70418_T901000_W7T": (50.0, 25.0),
        "70418_T901000_W8T": (75.0, 25.0),
    },
    # -----------------------------------------------------------
    # Name of the timestamp column.  None → auto-detect (first col)
    # -----------------------------------------------------------
    "timestamp_column": None,
    # -----------------------------------------------------------
    # Temperature unit label (displayed on plot axes)
    # -----------------------------------------------------------
    "temperature_unit": "°C",
    # -----------------------------------------------------------
    # Where to save exported HTML plots.  None → current directory
    # -----------------------------------------------------------
    "output_dir": None,
}

# ============================================================
#  RUN-MODE FLAGS
# ============================================================
RUN_TEST_MODE: bool = True   # True → synthetic data;  False → file dialog

# Which synthetic scenario to use when RUN_TEST_MODE is True
# Choices:  "traveling_hot_spot" | "thermal_ripple" | "pulsing_shockwave"
TEST_SCENARIO: str = "traveling_hot_spot"


# ============================================================
#  TEST DATA GENERATORS
# ============================================================

def _generate_test_timestamps(n_points: int = 300) -> pd.DatetimeIndex:
    """Return evenly-spaced timestamps spanning 5 minutes."""
    return pd.date_range("2024-01-01 08:00:00", periods=n_points, freq="1s")


def generate_traveling_hot_spot(
    config: Dict,
    n_points: int = 300,
) -> pd.DataFrame:
    """
    Scenario 1 – Traveling Hot Spot
    A localised hot spot moves across the spatial grid from left to right
    and then fades.  All thermocouples ride on a shared 800 °C baseline
    that drifts slowly upward.
    """
    rng = np.random.default_rng(42)
    timestamps = _generate_test_timestamps(n_points)
    tc_cols = config["thermocouple_columns"]
    spatial = config["spatial_map"]

    t = np.linspace(0, 1, n_points)
    baseline = 800.0 + 20.0 * t  # slow ramp

    data: Dict[str, np.ndarray] = {"Timestamp": timestamps}
    for col in tc_cols:
        x_tc, y_tc = spatial[col]
        # Hot-spot centre moves from x=0 to x=75 across the time window
        hot_x = 75.0 * t
        hot_y = 12.5  # mid-height
        dist = np.sqrt((x_tc - hot_x) ** 2 + (y_tc - hot_y) ** 2)
        spike = 40.0 * np.exp(-dist**2 / (2 * 15**2))  # Gaussian plume
        noise = rng.normal(0, 0.5, n_points)
        data[col] = baseline + spike + noise

    return pd.DataFrame(data)


def generate_thermal_ripple(
    config: Dict,
    n_points: int = 300,
) -> pd.DataFrame:
    """
    Scenario 2 – Thermal Ripple
    Sinusoidal temperature waves of different frequencies propagate through
    each thermocouple, creating a ripple effect across the sensor array.
    """
    rng = np.random.default_rng(7)
    timestamps = _generate_test_timestamps(n_points)
    tc_cols = config["thermocouple_columns"]
    spatial = config["spatial_map"]

    t = np.linspace(0, 2 * np.pi, n_points)
    baseline = 750.0

    data: Dict[str, np.ndarray] = {"Timestamp": timestamps}
    for i, col in enumerate(tc_cols):
        x_tc, _ = spatial[col]
        phase = x_tc / 75.0 * np.pi  # phase shift proportional to x-position
        ripple = 15.0 * np.sin(t - phase) + 8.0 * np.sin(2 * t - phase)
        noise = rng.normal(0, 0.4, n_points)
        data[col] = baseline + ripple + noise

    return pd.DataFrame(data)


def generate_pulsing_shockwave(
    config: Dict,
    n_points: int = 300,
) -> pd.DataFrame:
    """
    Scenario 3 – Pulsing Shockwave
    Periodic thermal shockwaves emanate from the control thermocouple
    (origin) and propagate outward, arriving at each sensor with a delay
    proportional to its distance from the origin.
    """
    rng = np.random.default_rng(99)
    timestamps = _generate_test_timestamps(n_points)
    tc_cols = config["thermocouple_columns"]
    spatial = config["spatial_map"]

    t = np.arange(n_points, dtype=float)
    pulse_period = 60  # samples between pulses
    baseline = 780.0
    wave_speed = 5.0  # samples per unit distance

    data: Dict[str, np.ndarray] = {"Timestamp": timestamps}
    for col in tc_cols:
        x_tc, y_tc = spatial[col]
        dist = np.sqrt(x_tc**2 + y_tc**2)
        delay = dist / wave_speed
        signal = np.zeros(n_points)
        for pulse_t in np.arange(0, n_points, pulse_period):
            arrival = pulse_t + delay
            signal += 30.0 * np.exp(-((t - arrival) ** 2) / (2 * 8**2))
        noise = rng.normal(0, 0.3, n_points)
        data[col] = baseline + signal + noise

    return pd.DataFrame(data)


_SCENARIO_MAP = {
    "traveling_hot_spot": generate_traveling_hot_spot,
    "thermal_ripple": generate_thermal_ripple,
    "pulsing_shockwave": generate_pulsing_shockwave,
}


# ============================================================
#  MAIN ANALYSER CLASS
# ============================================================

class ThermocoupleAnalyzer:
    """
    End-to-end thermocouple data loader, analyser, and visualiser.

    Parameters
    ----------
    config : dict
        Configuration dictionary (see USER_CONFIG at the top of this file).
    """

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.drift_df: Optional[pd.DataFrame] = None

        # Resolve output directory
        out = config.get("output_dir")
        self.output_dir = Path(out) if out else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_timestamp(series: pd.Series) -> pd.Series:
        """
        Robustly parse a mixed-format timestamp column.

        Tries, in order:
        1. pandas ``to_datetime`` automatic format detection
        2. ISO 8601 with and without fractional seconds
        3. Numeric (Unix epoch seconds / milliseconds)
        Falls back to leaving the original values if all attempts fail.
        """
        # Already datetime?
        if pd.api.types.is_datetime64_any_dtype(series):
            return series

        # Try the standard pandas parser first
        try:
            return pd.to_datetime(series)
        except (ValueError, TypeError):
            pass

        # Try common explicit formats
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
        ):
            try:
                return pd.to_datetime(series, format=fmt)
            except (ValueError, TypeError):
                continue

        # Numeric epoch?
        try:
            numeric = pd.to_numeric(series)
            # Heuristic: values > 1e10 are likely milliseconds
            if numeric.median() > 1e10:
                return pd.to_datetime(numeric, unit="ms")
            return pd.to_datetime(numeric, unit="s")
        except (ValueError, TypeError):
            pass

        warnings.warn(
            "Could not parse timestamp column – leaving as-is.", stacklevel=2
        )
        return series

    @staticmethod
    def _find_header_row(filepath: str) -> int:
        """
        Scan the first 30 rows of a CSV file to find the row that is most
        likely the header (contains the most non-numeric tokens).
        Returns 0 if no clear header is found.
        """
        best_row, best_score = 0, -1
        try:
            with open(filepath, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if i >= 30:
                        break
                    tokens = line.strip().split(",")
                    n_text = sum(
                        1
                        for tok in tokens
                        if not tok.strip().replace(".", "").replace("-", "").isdigit()
                        and tok.strip()
                    )
                    if n_text > best_score:
                        best_score, best_row = n_text, i
        except OSError:
            pass
        return best_row

    def _load_single_file(self, filepath: str) -> pd.DataFrame:
        """Load a single CSV or Excel file into a DataFrame."""
        ext = Path(filepath).suffix.lower()

        if ext in (".xls", ".xlsx", ".xlsm"):
            raw = pd.read_excel(filepath, header=None)
            # Find header row by looking for the row with the most string values
            best_row, best_score = 0, -1
            for i, row in raw.iterrows():
                n_text = sum(
                    1
                    for v in row
                    if isinstance(v, str)
                    and not v.strip().replace(".", "").replace("-", "").isdigit()
                    and v.strip()
                )
                if n_text > best_score:
                    best_score, best_row = n_text, i
                if i >= 30:
                    break
            df = pd.read_excel(filepath, header=int(best_row))
        else:
            header_row = self._find_header_row(filepath)
            df = pd.read_csv(
                filepath,
                header=header_row,
                encoding="utf-8",
                encoding_errors="replace",
            )

        # Strip whitespace from column names
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def load_data(self, filepaths: List[str]) -> pd.DataFrame:
        """
        Load one or more CSV / Excel files and concatenate them.

        The timestamp column is auto-detected (or taken from config) and
        parsed robustly.  Missing thermocouple columns are reported and
        skipped.

        Parameters
        ----------
        filepaths : list of str
            Paths to the data files to load.

        Returns
        -------
        pd.DataFrame
            Concatenated, time-sorted DataFrame with parsed timestamps.
        """
        frames: List[pd.DataFrame] = []
        for fp in filepaths:
            print(f"  Loading: {fp}")
            try:
                df = self._load_single_file(fp)
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Could not load {fp}: {exc}", stacklevel=2)

        if not frames:
            raise ValueError("No data files could be loaded.")

        combined = pd.concat(frames, ignore_index=True)

        # Identify timestamp column
        ts_col = self.config.get("timestamp_column")
        if ts_col is None:
            ts_col = combined.columns[0]
        if ts_col not in combined.columns:
            raise KeyError(
                f"Timestamp column '{ts_col}' not found in data. "
                f"Available columns: {list(combined.columns)}"
            )

        combined[ts_col] = self._parse_timestamp(combined[ts_col])
        combined = combined.sort_values(ts_col).reset_index(drop=True)
        combined = combined.rename(columns={ts_col: "Timestamp"})

        # Validate thermocouple columns
        tc_cols = self.config["thermocouple_columns"]
        available = [c for c in tc_cols if c in combined.columns]
        missing = [c for c in tc_cols if c not in combined.columns]
        if missing:
            warnings.warn(
                f"The following TC columns were not found and will be skipped: "
                f"{missing}",
                stacklevel=2,
            )
        if not available:
            raise ValueError(
                "None of the configured thermocouple columns were found in the data."
            )

        # Coerce TC columns to numeric
        for col in available:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

        self.df = combined
        print(
            f"  Loaded {len(combined):,} rows × {len(available)} thermocouples."
        )
        return combined

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def perform_advanced_analysis(self) -> pd.DataFrame:
        """
        Calculate per-thermocouple drift relative to the control channel.

        Drift is defined as:
            drift_i(t) = T_i(t) − T_control(t)

        The result is stored in ``self.drift_df`` and returned.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'Timestamp' and one column per thermocouple
            containing the drift values.
        """
        if self.df is None:
            raise RuntimeError("No data loaded.  Call load_data() first.")

        control = self.config["control_tc"]
        tc_cols = self._available_tc_columns()

        if control not in self.df.columns:
            raise KeyError(
                f"Control thermocouple '{control}' not found in data."
            )

        drift = pd.DataFrame({"Timestamp": self.df["Timestamp"]})
        for col in tc_cols:
            drift[col] = self.df[col] - self.df[control]

        self.drift_df = drift
        return drift

    # ------------------------------------------------------------------
    # Plot 1 – Raw temperatures
    # ------------------------------------------------------------------

    def plot_raw_temperatures(self, save_html: bool = True) -> go.Figure:
        """
        Interactive Plotly time-series of all raw thermocouple temperatures.

        Features
        --------
        * Custom legend labels and colours
        * Hover showing exact timestamp, label, and temperature
        * Range-selector buttons (1 min / 5 min / All)
        * Branded layout

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if self.df is None:
            raise RuntimeError("No data loaded.  Call load_data() first.")

        tc_cols = self._available_tc_columns()
        labels = self._legend_labels_for(tc_cols)
        colors = self._colors_for(tc_cols)
        unit = self.config["temperature_unit"]

        fig = go.Figure()
        for col, label, color in zip(tc_cols, labels, colors):
            fig.add_trace(
                go.Scatter(
                    x=self.df["Timestamp"],
                    y=self.df[col],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1.5),
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "Time: %{x|%Y-%m-%d %H:%M:%S}<br>"
                        f"Temp: %{{y:.2f}} {unit}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=dict(
                text="<b>Thermocouple Raw Temperatures</b>",
                font=dict(size=20),
            ),
            xaxis=dict(
                title="Time",
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1m", step="minute", stepmode="backward"),
                        dict(count=5, label="5m", step="minute", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                ),
            ),
            yaxis=dict(title=f"Temperature ({unit})"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            hovermode="x unified",
            template="plotly_white",
            height=600,
        )

        if save_html:
            path = self.output_dir / "raw_temperatures.html"
            fig.write_html(str(path))
            print(f"  Saved: {path}")

        _show_fig(fig)
        return fig

    # ------------------------------------------------------------------
    # Plot 2 – Drift analysis
    # ------------------------------------------------------------------

    def plot_drift(self, save_html: bool = True) -> go.Figure:
        """
        Interactive Plotly time-series of thermocouple drift
        (deviation from the control channel).

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if self.drift_df is None:
            self.perform_advanced_analysis()

        tc_cols = self._available_tc_columns()
        labels = self._legend_labels_for(tc_cols)
        colors = self._colors_for(tc_cols)
        control = self.config["control_tc"]
        unit = self.config["temperature_unit"]
        control_label = self._label_for(control)

        fig = go.Figure()
        for col, label, color in zip(tc_cols, labels, colors):
            if col == control:
                continue  # drift of control is identically 0 – skip
            fig.add_trace(
                go.Scatter(
                    x=self.drift_df["Timestamp"],
                    y=self.drift_df[col],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1.5),
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "Time: %{x|%Y-%m-%d %H:%M:%S}<br>"
                        f"Drift: %{{y:+.2f}} {unit}<extra></extra>"
                    ),
                )
            )

        # Zero-drift reference line
        fig.add_hline(
            y=0,
            line=dict(color="black", width=1, dash="dash"),
            annotation_text=f"Control: {control_label}",
            annotation_position="top right",
        )

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>Thermocouple Drift</b>  "
                    f"<sup>(relative to {control_label})</sup>"
                ),
                font=dict(size=20),
            ),
            xaxis=dict(
                title="Time",
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1m", step="minute", stepmode="backward"),
                        dict(count=5, label="5m", step="minute", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                ),
            ),
            yaxis=dict(title=f"Drift ({unit})"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            hovermode="x unified",
            template="plotly_white",
            height=600,
        )

        if save_html:
            path = self.output_dir / "drift_analysis.html"
            fig.write_html(str(path))
            print(f"  Saved: {path}")

        _show_fig(fig)
        return fig

    # ------------------------------------------------------------------
    # Plot 3 – Static contour (average drift heat-map)
    # ------------------------------------------------------------------

    def plot_contour(self, save_html: bool = True) -> go.Figure:
        """
        Static 2-D interpolated heat map of average temperature deviation
        across the spatial sensor layout.

        Uses ``scipy.interpolate.griddata`` (cubic method where possible,
        falling back to linear) to fill the grid between sensor positions.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if self.drift_df is None:
            self.perform_advanced_analysis()

        spatial = self.config["spatial_map"]
        tc_cols = self._available_tc_columns()
        unit = self.config["temperature_unit"]

        xs, ys, zs = [], [], []
        for col in tc_cols:
            if col not in spatial:
                continue
            x, y = spatial[col]
            mean_drift = float(self.drift_df[col].mean())
            xs.append(x)
            ys.append(y)
            zs.append(mean_drift)

        if len(xs) < 3:
            warnings.warn(
                "Not enough spatial points for contour interpolation (need ≥ 3).",
                stacklevel=2,
            )
            fig = go.Figure()
            _show_fig(fig)
            return fig

        xs_arr = np.array(xs, dtype=float)
        ys_arr = np.array(ys, dtype=float)
        zs_arr = np.array(zs, dtype=float)

        # Build interpolation grid
        margin = 5.0
        grid_x, grid_y = np.meshgrid(
            np.linspace(xs_arr.min() - margin, xs_arr.max() + margin, 200),
            np.linspace(ys_arr.min() - margin, ys_arr.max() + margin, 200),
        )
        points = np.column_stack([xs_arr, ys_arr])

        try:
            grid_z = griddata(points, zs_arr, (grid_x, grid_y), method="cubic")
            # Fill NaN edges with linear
            mask = np.isnan(grid_z)
            if mask.any():
                grid_z_linear = griddata(
                    points, zs_arr, (grid_x, grid_y), method="linear"
                )
                grid_z[mask] = grid_z_linear[mask]
        except Exception:  # noqa: BLE001
            grid_z = griddata(points, zs_arr, (grid_x, grid_y), method="linear")

        fig = go.Figure()

        # Filled contour
        fig.add_trace(
            go.Contour(
                x=grid_x[0],
                y=grid_y[:, 0],
                z=grid_z,
                colorscale="RdBu_r",
                colorbar=dict(title=f"Avg Drift ({unit})"),
                contours=dict(showlabels=True, labelfont=dict(size=10)),
                hovertemplate=(
                    "X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>"
                    f"Avg Drift: %{{z:.2f}} {unit}<extra></extra>"
                ),
            )
        )

        # Sensor markers
        labels = self._legend_labels_for(tc_cols)
        for col, x, y, z, label in zip(tc_cols, xs, ys, zs, labels):
            if col not in spatial:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    marker=dict(size=12, color="black", symbol="circle"),
                    text=[f"{label}<br>{z:+.1f} {unit}"],
                    textposition="top center",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        f"Position: ({x:.0f}, {y:.0f}) mm<br>"
                        f"Avg Drift: {z:+.2f} {unit}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=dict(
                text="<b>Spatial Heat Map – Average Temperature Drift</b>",
                font=dict(size=20),
            ),
            xaxis=dict(title="X Position (mm)", scaleanchor="y"),
            yaxis=dict(title="Y Position (mm)"),
            template="plotly_white",
            height=600,
        )

        if save_html:
            path = self.output_dir / "static_contour.html"
            fig.write_html(str(path))
            print(f"  Saved: {path}")

        _show_fig(fig)
        return fig

    # ------------------------------------------------------------------
    # Plot 4 – Dynamic animated contour
    # ------------------------------------------------------------------

    def plot_dynamic_contour(
        self,
        n_frames: Optional[int] = None,
        save_html: bool = True,
    ) -> go.Figure:
        """
        Animated 2-D contour movie showing how spatial temperature drift
        evolves over time.

        The user is prompted to choose an animation quality level (more
        frames = smoother but slower to render).

        Parameters
        ----------
        n_frames : int, optional
            Override the number of animation frames.  When ``None`` the
            user is prompted interactively.
        save_html : bool
            Whether to write the plot to an HTML file.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if self.drift_df is None:
            self.perform_advanced_analysis()

        spatial = self.config["spatial_map"]
        tc_cols = [c for c in self._available_tc_columns() if c in spatial]
        unit = self.config["temperature_unit"]

        if len(tc_cols) < 3:
            warnings.warn(
                "Not enough spatial points for dynamic contour (need ≥ 3).",
                stacklevel=2,
            )
            return go.Figure()

        # ------------------------------------------------------------------
        # Animation quality prompt
        # ------------------------------------------------------------------
        quality_options = {
            "1": ("Bare Minimum (Fastest) 🐢", 20),
            "2": ("Economy Class ✈️", 40),
            "3": ("Standard (Balanced) ⚖️", 80),
            "4": ("High Definition 🎬", 150),
            "5": ("Michael Jackson (Smoothest) 🕺", 300),
        }

        if n_frames is None:
            print("\n" + "=" * 55)
            print("  Dynamic Contour – Choose Animation Quality")
            print("=" * 55)
            for key, (name, nf) in quality_options.items():
                print(f"  [{key}]  {name:35s}  ({nf} frames)")
            print("=" * 55)
            choice = input("  Enter choice [1-5] (default 3): ").strip() or "3"
            _, n_frames = quality_options.get(choice, quality_options["3"])
            print(f"  → Rendering {n_frames} frames …\n")

        total_rows = len(self.drift_df)
        step = max(1, total_rows // n_frames)
        frame_indices = list(range(0, total_rows, step))[:n_frames]

        # Spatial grid
        xs_arr = np.array([spatial[c][0] for c in tc_cols], dtype=float)
        ys_arr = np.array([spatial[c][1] for c in tc_cols], dtype=float)
        points = np.column_stack([xs_arr, ys_arr])
        margin = 5.0
        grid_x, grid_y = np.meshgrid(
            np.linspace(xs_arr.min() - margin, xs_arr.max() + margin, 80),
            np.linspace(ys_arr.min() - margin, ys_arr.max() + margin, 80),
        )

        def _interp(idx: int) -> np.ndarray:
            row = self.drift_df.iloc[idx]
            zs = np.array([float(row[c]) for c in tc_cols])
            gz = griddata(points, zs, (grid_x, grid_y), method="linear")
            return gz

        # Pre-compute z-range for consistent colour scale
        sample_rows = frame_indices[:: max(1, len(frame_indices) // 10)]
        all_z: List[float] = []
        for idx in sample_rows:
            gz = _interp(idx)
            valid = gz[~np.isnan(gz)]
            if valid.size:
                all_z.extend(valid.tolist())

        zmin = min(all_z) if all_z else -10.0
        zmax = max(all_z) if all_z else 10.0
        sym = max(abs(zmin), abs(zmax))
        zmin, zmax = -sym, sym

        # ------------------------------------------------------------------
        # Build frames
        # ------------------------------------------------------------------
        frames: List[go.Frame] = []
        for frame_num, idx in enumerate(frame_indices):
            gz = _interp(idx)
            ts = self.drift_df["Timestamp"].iloc[idx]
            ts_str = (
                ts.strftime("%Y-%m-%d %H:%M:%S")
                if hasattr(ts, "strftime")
                else str(ts)
            )
            frame = go.Frame(
                data=[
                    go.Contour(
                        x=grid_x[0],
                        y=grid_y[:, 0],
                        z=gz,
                        colorscale="RdBu_r",
                        zmin=zmin,
                        zmax=zmax,
                        showscale=True,
                        colorbar=dict(title=f"Drift ({unit})"),
                        contours=dict(showlabels=False),
                        hovertemplate=(
                            "X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>"
                            f"Drift: %{{z:.2f}} {unit}<extra></extra>"
                        ),
                    )
                ],
                layout=go.Layout(
                    title_text=(
                        f"<b>Dynamic Thermal Map</b>  "
                        f"<sup>{ts_str}  |  frame {frame_num + 1}/{len(frame_indices)}</sup>"
                    )
                ),
                name=str(frame_num),
            )
            frames.append(frame)

        # ------------------------------------------------------------------
        # Initial contour (first frame)
        # ------------------------------------------------------------------
        gz0 = _interp(frame_indices[0])
        fig = go.Figure(
            data=[
                go.Contour(
                    x=grid_x[0],
                    y=grid_y[:, 0],
                    z=gz0,
                    colorscale="RdBu_r",
                    zmin=zmin,
                    zmax=zmax,
                    showscale=True,
                    colorbar=dict(title=f"Drift ({unit})"),
                )
            ],
            frames=frames,
        )

        # Sensor position markers (static overlay)
        labels = self._legend_labels_for(tc_cols)
        fig.add_trace(
            go.Scatter(
                x=xs_arr,
                y=ys_arr,
                mode="markers+text",
                marker=dict(size=10, color="black", symbol="circle"),
                text=labels,
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Slider steps
        slider_steps = [
            dict(
                args=[
                    [str(k)],
                    dict(
                        frame=dict(duration=50, redraw=True),
                        mode="immediate",
                        transition=dict(duration=0),
                    ),
                ],
                label=str(k),
                method="animate",
            )
            for k in range(len(frames))
        ]

        fig.update_layout(
            title=dict(
                text="<b>Dynamic Thermal Map</b>",
                font=dict(size=20),
            ),
            xaxis=dict(title="X Position (mm)", scaleanchor="y"),
            yaxis=dict(title="Y Position (mm)"),
            template="plotly_white",
            height=650,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.08,
                    x=0.0,
                    xanchor="left",
                    buttons=[
                        dict(
                            label="▶  Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=80, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="⏸  Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    steps=slider_steps,
                    active=0,
                    currentvalue=dict(
                        prefix="Frame: ",
                        font=dict(size=13),
                        visible=True,
                    ),
                    pad=dict(t=50),
                    len=0.9,
                    x=0.05,
                )
            ],
        )

        if save_html:
            path = self.output_dir / "dynamic_contour.html"
            fig.write_html(str(path))
            print(f"  Saved: {path}")

        _show_fig(fig)
        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _available_tc_columns(self) -> List[str]:
        """Return configured TC columns that are actually present in the data."""
        if self.df is None:
            return list(self.config["thermocouple_columns"])
        return [
            c
            for c in self.config["thermocouple_columns"]
            if c in self.df.columns
        ]

    def _legend_labels_for(self, columns: List[str]) -> List[str]:
        """Return the legend label for each column (falls back to column name)."""
        col_to_label = dict(
            zip(
                self.config["thermocouple_columns"],
                self.config.get("legend_labels", []),
            )
        )
        return [col_to_label.get(c, c) for c in columns]

    def _label_for(self, column: str) -> str:
        """Return the legend label for a single column."""
        return self._legend_labels_for([column])[0]

    def _colors_for(self, columns: List[str]) -> List[str]:
        """Return the colour for each column (cycles if more columns than colours)."""
        palette = self.config.get("colors", px.colors.qualitative.Plotly)
        col_to_color = {
            c: palette[i % len(palette)]
            for i, c in enumerate(self.config["thermocouple_columns"])
        }
        return [col_to_color.get(c, palette[0]) for c in columns]


# ============================================================
#  ENTRY POINT
# ============================================================

def _pick_files_via_dialog() -> List[str]:
    """
    Open a tkinter file-picker dialog for selecting data files.
    Falls back gracefully if a display is not available.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.lift()
        paths = filedialog.askopenfilenames(
            title="Select Thermocouple Data Files",
            filetypes=[
                ("Supported files", "*.csv *.xlsx *.xls *.xlsm"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls *.xlsm"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return list(paths)
    except Exception as exc:  # noqa: BLE001
        print(f"  tkinter file dialog unavailable ({exc}).")
        path = input("  Enter the path to your data file: ").strip()
        return [path] if path else []


def main() -> None:
    """Main entry-point."""
    print("=" * 60)
    print("  Thermocouple Analysis & Visualization Tool")
    print("=" * 60)

    analyzer = ThermocoupleAnalyzer(USER_CONFIG)

    if RUN_TEST_MODE:
        # -----------------------------------------------------------------
        # Synthetic test data
        # -----------------------------------------------------------------
        generator = _SCENARIO_MAP.get(TEST_SCENARIO)
        if generator is None:
            raise ValueError(
                f"Unknown TEST_SCENARIO '{TEST_SCENARIO}'. "
                f"Choose from: {list(_SCENARIO_MAP.keys())}"
            )
        print(f"\n  [TEST MODE]  Scenario: {TEST_SCENARIO}")
        df = generator(USER_CONFIG)
        analyzer.df = df
        tc_cols = USER_CONFIG["thermocouple_columns"]
        for col in tc_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"  Generated {len(df):,} synthetic data points.\n")
    else:
        # -----------------------------------------------------------------
        # Real data files
        # -----------------------------------------------------------------
        print("\n  Please select your data file(s) …")
        filepaths = _pick_files_via_dialog()
        if not filepaths:
            print("  No files selected.  Exiting.")
            return
        print()
        analyzer.load_data(filepaths)
        print()

    # Run analysis
    print("  Running drift analysis …")
    analyzer.perform_advanced_analysis()
    print("  Done.\n")

    # Plot 1 – Raw temperatures
    print("  Generating Plot 1 of 4: Raw Temperatures …")
    analyzer.plot_raw_temperatures()

    # Plot 2 – Drift
    print("  Generating Plot 2 of 4: Drift Analysis …")
    analyzer.plot_drift()

    # Plot 3 – Static contour
    print("  Generating Plot 3 of 4: Static Spatial Contour …")
    analyzer.plot_contour()

    # Plot 4 – Dynamic animated contour
    print("  Generating Plot 4 of 4: Dynamic Animated Contour …")
    analyzer.plot_dynamic_contour()

    print("\n  All plots generated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
