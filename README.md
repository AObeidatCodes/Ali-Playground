# Ali-Playground

## Thermocouple Analysis & Visualization Tool

A comprehensive, interactive tool for analysing time-series temperature data
from thermocouple test benches (e.g. SOFC/SOEC rigs).

---

### Features

| Feature | Description |
|---|---|
| **Flexible data loading** | CSV and Excel (`.csv`, `.xlsx`, `.xls`, `.xlsm`), auto-detects header row, handles mixed timestamp formats, concatenates multiple files |
| **Drift analysis** | Per-channel deviation from a configurable control thermocouple |
| **Raw temperature plot** | Interactive Plotly time-series with range-selector, hover, and custom colour/legend |
| **Drift plot** | Time-series of drift values, zero-reference line |
| **Static contour** | Scipy-interpolated 2-D heat map of average spatial drift |
| **Animated contour** | Frame-by-frame movie of spatial heat flow with Play/Pause and time-slider |
| **Test mode** | Three built-in synthetic scenarios for demo/debugging without real data |
| **User config** | Single `USER_CONFIG` dict – no class-level changes required |

---

### Requirements

```
pip install plotly scipy pandas numpy openpyxl
```

> `tkinter` is part of the Python standard library (used for the file dialog).
> On headless Linux servers run with `RUN_TEST_MODE = True`.

---

### Quick Start

**Demo mode (no data files needed)**

```bash
python thermocouple_analyzer.py
```

`RUN_TEST_MODE = True` is the default.  Pick a scenario by editing:

```python
TEST_SCENARIO = "traveling_hot_spot"   # or "thermal_ripple" / "pulsing_shockwave"
```

**Real data**

1. Set `RUN_TEST_MODE = False` in the script.
2. Edit `USER_CONFIG` to match your channel names, legend labels, colours,
   and spatial coordinates.
3. Run the script – a file dialog opens to select one or more CSV/Excel files.

```bash
python thermocouple_analyzer.py
```

Four interactive HTML plots are written to the working directory:

| File | Contents |
|---|---|
| `raw_temperatures.html` | All channels over time |
| `drift_analysis.html` | Per-channel drift from control |
| `static_contour.html` | Average drift heat map |
| `dynamic_contour.html` | Animated spatial heat movie |

---

### Configuration Reference (`USER_CONFIG`)

| Key | Type | Description |
|---|---|---|
| `thermocouple_columns` | `list[str]` | Column names in the data file |
| `legend_labels` | `list[str]` | Human-readable labels (same order) |
| `control_tc` | `str` | Reference channel for drift calculation |
| `colors` | `list[str]` | Plotly colour strings / hex codes |
| `spatial_map` | `dict` | `{column: (x_mm, y_mm)}` positions |
| `timestamp_column` | `str \| None` | Timestamp column name (`None` → auto) |
| `temperature_unit` | `str` | Axis label, e.g. `"°C"` |
| `output_dir` | `str \| None` | HTML output folder (`None` → cwd) |

---

### Test Scenarios

| Scenario | Description |
|---|---|
| `traveling_hot_spot` | Hot spot sweeps across the sensor array |
| `thermal_ripple` | Sinusoidal waves propagate with spatial phase shifts |
| `pulsing_shockwave` | Periodic shockwaves emanate from the origin sensor |

---

### Animation Quality Prompt

When `plot_dynamic_contour()` runs you are asked to choose a quality level:

```
[1]  Bare Minimum (Fastest) 🐢           (20 frames)
[2]  Economy Class ✈️                    (40 frames)
[3]  Standard (Balanced) ⚖️             (80 frames)
[4]  High Definition 🎬                 (150 frames)
[5]  Michael Jackson (Smoothest) 🕺      (300 frames)
```

Press **Enter** to accept the default (Standard).
