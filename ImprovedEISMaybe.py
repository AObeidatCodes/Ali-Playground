# %%
import os
import glob
import re
import datetime
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import tkinter as tk
from tkinter import filedialog, messagebox

pio.renderers.default = "notebook"

# %%
class SOECFootprintAnalyzer:
    # =========================================================================
    # 1. INITIALIZATION
    # =========================================================================
    
    # Baseline (reference) flow setpoints per variation type and the matching metadata.
    # Adding a new flow-variation type is as simple as adding an entry here.
    FLOW_SPECS = {
        'Airflow Variation': {
            'setpoint_col': 'setpoint_air',
            'baseline': 1400,
            'sweep_order': [1400, 1000, 600, 200, 100],
            'return_anchor': 1000,
            'flow_label': 'Airflow',
            'unit_label': 'Air',
            'x_title': 'Luftdurchfluss (mL/min)',
            'next_step_keys': ('airflow variation', 'luftdurchfluss'),
        },
        'Fuel Flow Variation': {
            'setpoint_col': 'setpoint_h2',
            'baseline': 660,
            'sweep_order': [660, 500, 400, 300, 100],
            'return_anchor': 500,
            'flow_label': 'Fuel Flow',
            'unit_label': 'Fuel',
            'x_title': 'Brenngasfluss (mL/min)',
            'next_step_keys': ('variation in fuelflow', 'variation fuelflow', 'brenngas'),
        },
         # --- NEW H2/N2 PHASES ---
        'H2/N2 Variation 0% H2O': {
            'next_step_keys': ('h2/n2 variation at 0% h2o', 'h2/n2 variation 0%'),
        },
        'H2/N2 Variation 10% H2O': {
            'next_step_keys': ('h2/n2 variation at 10% h2o', 'h2/n2 variation 10%'),
        },
        # ------------------------
    }

    def __init__(self, folders, config):
        self.folders = folders
        self.config = config
        
        self.cell_id = "Unknown_Cell"
        self.test_name = "Unknown_Test"
        self.cell_area = 1.0
        
        self.timeline = []
        self.eis_data = []
        self.figures = {}
        
        # Multiplier to ensure we don't run out of colors for large datasets
        self.colors = self.config.get("custom_colors", px.colors.qualitative.Plotly) * 3

    # =========================================================================
    # 2. MAIN EXECUTION PIPELINE
    # =========================================================================
    
    def parse_log_file(self):
        print("\n--- 1. Semantic Deep Dive: Parsing Master Log(s) ---")
        
        timestamp_pattern = re.compile(r'^.*?(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})|(\d{2}\.\d{2}\.\d{4}\s\d{2}:\d{2}:\d{2})')
        area_pattern = re.compile(r'Cell\s*area\s*:\s*([\d,.]+)', re.IGNORECASE)
        cell_name_pattern = re.compile(r'Cell-Name\s*:\s*([A-Za-z0-9_-]+)', re.IGNORECASE)
        test_name_pattern = re.compile(r'Test-Name:\s*(.*)', re.IGNORECASE)
        
        # Added "Start Variation" to catch testbench 70585 phase changes
        phase_pattern = re.compile(r'Next step:\s*(.*)|Start Variation:\s*(.*)', re.IGNORECASE)
        
        # HIGH PRECISION REGEXES (from inside Parameter_Info)
        exact_voltage_pattern = re.compile(r'voltage:\s*([\d,.]+)', re.IGNORECASE)
        exact_h2_pattern = re.compile(r'Anode-Wasserstoff \(H2\):\s*([\d,.]+)\s*ml/min\s*([\d,.]+)\s*ml/min', re.IGNORECASE)
        exact_o2_pattern = re.compile(r'Anode-Sauerstoff \(O2\):\s*([\d,.]+)\s*ml/min', re.IGNORECASE)
        exact_air_pattern = re.compile(r'Kathode-Luft:\s*([\d,.]+)\s*(?:ml/min)?\s*([\d,.]+)\s*(?:ml/min)?', re.IGNORECASE)
        exact_temp_pattern = re.compile(r'Temperature:\s*[\d,.]+°C\s+([\d,.]+)°C', re.IGNORECASE)
        
        # Legacy extractors
        voltage_pattern = re.compile(r'voltage:\s*([\d,.]+)', re.IGNORECASE)
        h2_flow_pattern = re.compile(r'\(H2\):\s*([\d,.]+)', re.IGNORECASE)
        
        # EXPANDED: Temperature and Humidity capture for both old testbench and new 70585
        temp_pattern = re.compile(r'set T901000 temperature to ([\d.,]+)|Target Temperature:\s*([\d.,]+)|Temperature:\s*([\d.,]+)°C|T=([\d.,]+)°C|Temperature is now at ([\d.,]+)°C', re.IGNORECASE)
        hum_pattern = re.compile(r'pH2O\s*=\s*(\d+)%|Gaszusammensetzung:\s*(\d+)%\s*H2O|(\d+)%\s*H2O\s*humidity', re.IGNORECASE)
        
        current_temp, current_humidity, current_flow, current_ocv = 650, 0, "High Flow", 0.0
        current_phase = "Standard"
        in_param_info = False
        exact_ocv = 0.0
        exact_temp = 650.0
        actual_h2, setpoint_h2 = 660.0, 660.0
        actual_air, setpoint_air = 1400.0, 1400.0
        actual_h2o = 0.0
        
        timeline_dict = {}
        for folder in self.folders:
            log_files = glob.glob(os.path.join(folder, "*LogFile*.txt")) + glob.glob(os.path.join(folder, "*.log"))
            if not log_files:
                print(f"  -> WARNING: No log file found in {folder}")
                continue
            
            log_path = log_files[0]
            print(f"  -> Found log file: {os.path.basename(log_path)}")
            
            try:
                with open(log_path, 'r', encoding='latin1') as f:
                    content = f.read()
                    cell_match = cell_name_pattern.search(content)
                    area_match = area_pattern.search(content)
                    test_name_match = test_name_pattern.search(content)
                    if cell_match and self.cell_id == "Unknown_Cell":
                        self.cell_id = cell_match.group(1).strip()
                    if area_match and self.cell_area == 1.0:
                        self.cell_area = float(area_match.group(1).replace(',', '.'))
                    if test_name_match and self.test_name == "Unknown_Test":
                        self.test_name = test_name_match.group(1).strip()
            except Exception:
                pass
                
            print(f"  -> Extracted Cell ID: {self.cell_id}")
            print(f"  -> Extracted Test Name: {self.test_name}")
            print(f"  -> Extracted Cell Area: {self.cell_area:.3f} cm²")
            
            with open(log_path, 'r', encoding='latin1') as f:
                for line in f:
                    # --- Phase tracking ---
                    phase_match = phase_pattern.search(line)
                    if phase_match:
                        step_text = (phase_match.group(1) or phase_match.group(2) or "").lower()
                        matched_flow = False
                        for phase_name, spec in self.FLOW_SPECS.items():
                            if any(key in step_text for key in spec['next_step_keys']):
                                current_phase = phase_name
                                matched_flow = True
                                break
                        if not matched_flow:
                            current_phase = "Standard"
                            
                    if "Sub Parameter_Info, START" in line:
                        in_param_info = True
                    elif "Sub Parameter_Info, END" in line:
                        in_param_info = False
                        
                    if in_param_info:
                        v_match = exact_voltage_pattern.search(line)
                        if v_match:
                            exact_ocv = float(v_match.group(1).replace(',', '.'))
                            current_ocv = exact_ocv
                            
                        h2_ex_match = exact_h2_pattern.search(line)
                        if h2_ex_match:
                            a_h2 = float(h2_ex_match.group(1).replace(',', '.'))
                            s_h2 = float(h2_ex_match.group(2).replace(',', '.'))
                            # AUTO-CORRECT: Testbench 70585 often logs flow in L/min but labels it ml/min (e.g. 0,66). 
                            # If it's suspiciously low, auto-convert it to ml/min to preserve plotting.
                            if 0 < a_h2 < 10.0: a_h2 *= 1000.0
                            if 0 < s_h2 < 10.0: s_h2 *= 1000.0
                            actual_h2 = a_h2
                            setpoint_h2 = s_h2
                            
                        air_ex_match = exact_air_pattern.search(line)
                        if air_ex_match:
                            a_air = float(air_ex_match.group(1).replace(',', '.'))
                            s_air = float(air_ex_match.group(2).replace(',', '.'))
                            # Auto-convert L/min to ml/min
                            if 0 < a_air < 10.0: a_air *= 1000.0
                            if 0 < s_air < 10.0: s_air *= 1000.0
                            actual_air = a_air
                            setpoint_air = s_air
                            
                        t_ex_match = exact_temp_pattern.search(line)
                        if t_ex_match:
                            exact_temp = float(t_ex_match.group(1).replace(',', '.'))
                            
                        o2_ex_match = exact_o2_pattern.search(line)
                        if o2_ex_match:
                            o2_flow = float(o2_ex_match.group(1).replace(',', '.'))
                            if 0 < o2_flow < 10.0: o2_flow *= 1000.0
                            actual_h2o = o2_flow * 2
                            
                    if not in_param_info:
                        v_legacy = voltage_pattern.search(line)
                        if v_legacy:
                            current_ocv = float(v_legacy.group(1).replace(',', '.'))
                            
                    # Track Explicit "Low Flow" tags from 70585 markers
                    if "(Low Flow)" in line:
                        current_flow = "Low Flow"
                    elif "High Flow" in line:
                        current_flow = "High Flow"
                            
                    h2_match = h2_flow_pattern.search(line)
                    if h2_match:
                        val = float(h2_match.group(1).replace(',', '.'))
                        if 0 < val < 10.0: val *= 1000.0
                        current_flow = "High Flow" if val > 300 else "Low Flow"
                        
                    temp_match = temp_pattern.search(line)
                    if temp_match:
                        raw_t = next((g for g in temp_match.groups() if g is not None), None)
                        if raw_t:
                            # Round to nearest 5 to stabilize slight test stand fluctuations
                            current_temp = int(round(float(raw_t.replace(',', '.')) / 5.0) * 5.0)
                            
                    hum_match = hum_pattern.search(line)
                    if hum_match:
                        raw_h = next((g for g in hum_match.groups() if g is not None), None)
                        if raw_h:
                            current_humidity = int(raw_h)
                            
                    ts_match = timestamp_pattern.search(line)
                    if ts_match:
                        ts_str = ts_match.group(1) or ts_match.group(2)
                        if '-' in ts_str:
                            dt = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                        else:
                            dt = datetime.datetime.strptime(ts_str, "%d.%m.%Y %H:%M:%S")
                        
                        # Master Timeline Deduplication Dictionary
                        timeline_dict[dt] = {
                            'timestamp': dt,
                            'temperature': current_temp,
                            'humidity': current_humidity,
                            'flow_regime': current_flow,
                            'ocv': current_ocv,
                            'phase': current_phase,
                            'exact_ocv': exact_ocv,
                            'exact_temp': exact_temp,
                            'actual_h2': actual_h2,
                            'setpoint_h2': setpoint_h2,
                            'actual_air': actual_air,
                            'setpoint_air': setpoint_air,
                            'actual_h2o': actual_h2o,
                        }
                        
        self.timeline = list(timeline_dict.values())
        self.timeline.sort(key=lambda x: x['timestamp'])

    def parse_eis_files(self):
        print("\n--- 2. Parsing EIS Data & Mapping to Timeline ---")
        eis_files_raw = []
        for folder in self.folders:
            # Recursively search for both lowercase and uppercase EIS files
            found_lower = glob.glob(os.path.join(folder, "**", "eis_*.txt"), recursive=True)
            found_upper = glob.glob(os.path.join(folder, "**", "EIS_*.txt"), recursive=True)
            eis_files_raw.extend(found_lower)
            eis_files_raw.extend(found_upper)
                
        # Deduplicate files by filename case-insensitively
        unique_eis_files = {}
        for fpath in eis_files_raw:
            unique_eis_files[os.path.basename(fpath).lower()] = fpath
        eis_files = list(unique_eis_files.values())
        
        print(f"  -> Found {len(eis_files)} EIS file(s) across folders.")
        if not self.timeline:
            print("  -> ERROR: Timeline is empty. Cannot map EIS data.")
            return
            
        self.timeline.sort(key=lambda x: x['timestamp'])
        
        successful_parses = 0
        for file_path in eis_files:
            filename = os.path.basename(file_path)
            # Add re.IGNORECASE so it matches both eis_ and EIS_
            match = re.search(r'eis_(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', filename, re.IGNORECASE)
            if not match:
                continue
                
            yy, mm, dd, hr, mn, sc = match.groups()
            file_time = datetime.datetime(int("20" + yy), int(mm), int(dd), int(hr), int(mn), int(sc))
            
            current_state = self.timeline[0].copy()
            for event in self.timeline:
                if event['timestamp'] <= file_time:
                    current_state.update(event)
                else:
                    break
                    
            r_val, i_val, freqs = self.read_impedance_file(file_path)
            if r_val is not None and len(r_val) > 0:
                successful_parses += 1
                self.eis_data.append(
                    {
                        'filename': filename,
                        'timestamp': file_time,
                        **current_state,
                        'Re(Z)': r_val,
                        '-Im(Z)': i_val,
                        'Frequency': freqs,
                    }
                )
        
        print(f"  -> Successfully parsed data from {successful_parses} EIS file(s).")
                
        if not self.eis_data:
            print("  -> WARNING: No valid EIS files parsed. The files might not contain 'Impedance samples' headers.")
            self.df_files = pd.DataFrame()
        else:
            self.df_files = pd.DataFrame(self.eis_data).sort_values('timestamp')

    def generate_analyst_plots(self, preview_data=False, jc_mode=False):
        print("\n--- 3. Generating Analyst Footprint Sequences ---")
        if jc_mode:
            print("  -> JC Mode Active: Keeping 1:1 aspect ratio but cropping figure vertically...")
        if self.df_files.empty:
            print("No data available to plot.")
            return
        df = self.df_files
        
        # --- NERNST LEAK CURRENT CALCULATIONS ---
        if 'exact_ocv' in df.columns:
            df['leak_current'] = df.apply(
                lambda row: self.calculate_leak_current(
                    T_C=row['exact_temp'],
                    OCV=row['exact_ocv'],
                    vH2=row['actual_h2'],
                    vH2O=row['actual_h2o'],
                    A=self.cell_area,
                ),
                axis=1,
            )
            
        # DATA VALIDATION CONSOLE PRINT
        print("\n" + "=" * 80)
        print("🔍 RAW DATA EXTRACTED FOR LEAK CURRENT CALCULATIONS 🔍")
        print("=" * 80)
        check_df = df[df['phase'].isin(['Airflow Variation', 'Fuel Flow Variation'])]
        if not check_df.empty:
            view_cols = [
                'filename', 'phase', 'setpoint_air', 'setpoint_h2', 
                'exact_temp', 'exact_ocv', 'leak_current',
            ]
            print(check_df[view_cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        else:
            print("No 'Airflow Variation' or 'Fuel Flow Variation' phases detected in the log timeline.")
        print("=" * 80 + "\n")
        
        # --- 1. AIRFLOW VARIATION PLOTS ---
        air_df = df[df['phase'] == 'Airflow Variation'].sort_values('timestamp')
        if not air_df.empty:
            main_air_df = air_df.iloc[:-1]
            pt_1000 = air_df[air_df['setpoint_air'] == 1000].head(1)
            dashed_air_df = pd.concat([pt_1000, air_df.iloc[-1:]]) if not pt_1000.empty else pd.DataFrame()
            
            for view_mode in ["Zoomed Out", "Zoomed In"]:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                # Solid Lines
                fig.add_trace(
                    go.Scatter(
                        x=main_air_df['setpoint_air'], y=main_air_df['exact_ocv'], mode='lines+markers',
                        name='OCV', line=dict(color='#1f77b4', width=3), marker=dict(size=10),
                    ),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(
                        x=main_air_df['setpoint_air'], y=main_air_df['leak_current'], mode='lines+markers',
                        name='Leckstromdichte', line=dict(color='#d62728', width=3), marker=dict(size=10),
                    ),
                    secondary_y=True,
                )
                # Dashed Lines
                if not dashed_air_df.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=dashed_air_df['setpoint_air'], y=dashed_air_df['exact_ocv'], mode='lines+markers',
                            name='OCV (EoT Return)', line=dict(color='#1f77b4', width=2, dash='dash'),
                            marker=dict(size=12, symbol='star'),
                        ),
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=dashed_air_df['setpoint_air'], y=dashed_air_df['leak_current'], mode='lines+markers',
                            name='Leckstromdichte (EoT Return)', line=dict(color='#d62728', width=2, dash='dash'),
                            marker=dict(size=12, symbol='star'),
                        ),
                        secondary_y=True,
                    )
                # Annotations
                if not main_air_df.empty:
                    x0_air = main_air_df.iloc[0]['setpoint_air']
                    fig.add_annotation(
                        x=x0_air, y=main_air_df.iloc[0]['exact_ocv'], xref="x", yref="y",
                        axref="pixel", ayref="pixel", text="BoT", showarrow=True, arrowhead=2, ax=-25, ay=-35, font=dict(color="#1f77b4", size=16),
                    )
                    fig.add_annotation(
                        x=x0_air, y=main_air_df.iloc[0]['leak_current'], xref="x", yref="y2",
                        axref="pixel", ayref="pixel", text="BoT", showarrow=True, arrowhead=2, ax=-25, ay=35, font=dict(color="#d62728", size=16),
                    )
                if not dashed_air_df.empty:
                    xN_air = dashed_air_df.iloc[-1]['setpoint_air']
                    fig.add_annotation(
                        x=xN_air, y=dashed_air_df.iloc[-1]['exact_ocv'], xref="x", yref="y",
                        axref="pixel", ayref="pixel", text="EoT", showarrow=True, arrowhead=2, ax=30, ay=35, font=dict(color="#1f77b4", size=16),
                    )
                    fig.add_annotation(
                        x=xN_air, y=dashed_air_df.iloc[-1]['leak_current'], xref="x", yref="y2",
                        axref="pixel", ayref="pixel", text="EoT", showarrow=True, arrowhead=2, ax=30, ay=-35, font=dict(color="#d62728", size=16),
                    )
                plot_title = f"Airflow Variation for {self.cell_id}" if view_mode == "Zoomed Out" else f"Airflow Variation for {self.cell_id} (Zoomed In)"
                fig.update_layout(title=plot_title, template="plotly_white", width=1400, height=800)
                fig.update_xaxes(title_text="Luft durchfluss (in mL/min)")
                if view_mode == "Zoomed Out":
                    max_val = max(air_df['exact_ocv'].max(), air_df['leak_current'].max()) * 1.1
                    fig.update_yaxes(title_text="OCV [V]", secondary_y=False, range=[0, max_val], title_font=dict(color="#1f77b4"), tickfont=dict(color="#1f77b4"))
                    fig.update_yaxes(title_text="Leckstromdichte [A/cm²]", secondary_y=True, range=[0, max_val], title_font=dict(color="#d62728"), tickfont=dict(color="#d62728"))
                    self.figures[f"Airflow Variation Plot - {self.cell_id}"] = fig
                else:
                    fig.update_yaxes(title_text="OCV [V]", secondary_y=False, rangemode="normal", title_font=dict(color="#1f77b4"), tickfont=dict(color="#1f77b4"))
                    fig.update_yaxes(title_text="Leckstromdichte [A/cm²]", secondary_y=True, rangemode="normal", title_font=dict(color="#d62728"), tickfont=dict(color="#d62728"))
                    self.figures[f"Airflow Variation Plot - {self.cell_id} - Zoomed In"] = fig
            
            # Airflow EIS Variation & Aging
            eis_air_title = f"Airflow Variation of {self.cell_id} @{int(air_df.iloc[0]['exact_temp'])}°C and 100% H₂"
            fig_eis_air = self._create_base_figure(eis_air_title)
            for i, row in air_df.reset_index().iterrows():
                suffix = " (BoT)" if i == 0 else (" (EoT)" if i == len(air_df) - 1 else "")
                self._add_trace(fig_eis_air, row, f"{int(row['setpoint_air'])} mL/min Air{suffix}", i)
            self.figures[eis_air_title] = fig_eis_air
            
            air_1400 = air_df[air_df['setpoint_air'] == 1400]
            if len(air_1400) >= 2:
                title = f"Aging of {self.cell_id} after Airflow Variation"
                fig_aging_air = self._create_base_figure(title)
                self._add_trace(fig_aging_air, air_1400.iloc[0], f"BoT (1400 mL/min Air) OCV: {air_1400.iloc[0]['exact_ocv']:.3f}V", 0)
                self._add_trace(fig_aging_air, air_1400.iloc[-1], f"EoT (1400 mL/min Air) OCV: {air_1400.iloc[-1]['exact_ocv']:.3f}V", 1)
                self.figures[title] = fig_aging_air

        # --- 2. FUEL FLOW VARIATION PLOTS ---
        fuel_df = df[df['phase'] == 'Fuel Flow Variation'].sort_values('timestamp')
        if not fuel_df.empty:
            main_fuel_df = fuel_df.iloc[:-1]
            pt_500 = fuel_df[fuel_df['setpoint_h2'] == 500].head(1)
            dashed_fuel_df = pd.concat([pt_500, fuel_df.iloc[-1:]]) if not pt_500.empty else pd.DataFrame()
            
            for view_mode in ["Zoomed Out", "Zoomed In"]:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                # Solid Lines
                fig.add_trace(
                    go.Scatter(
                        x=main_fuel_df['setpoint_h2'], y=main_fuel_df['exact_ocv'], mode='lines+markers',
                        name='OCV', line=dict(color='#1f77b4', width=3), marker=dict(size=10),
                    ),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(
                        x=main_fuel_df['setpoint_h2'], y=main_fuel_df['leak_current'], mode='lines+markers',
                        name='Leckstromdichte', line=dict(color='#d62728', width=3), marker=dict(size=10),
                    ),
                    secondary_y=True,
                )
                # Dashed Lines
                if not dashed_fuel_df.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=dashed_fuel_df['setpoint_h2'], y=dashed_fuel_df['exact_ocv'], mode='lines+markers',
                            name='OCV (EoT Return)', line=dict(color='#1f77b4', width=2, dash='dash'),
                            marker=dict(size=12, symbol='star'),
                        ),
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=dashed_fuel_df['setpoint_h2'], y=dashed_fuel_df['leak_current'], mode='lines+markers',
                            name='Leckstromdichte (EoT Return)', line=dict(color='#d62728', width=2, dash='dash'),
                            marker=dict(size=12, symbol='star'),
                        ),
                        secondary_y=True,
                    )
                # Annotations
                if not main_fuel_df.empty:
                    x0_fuel = main_fuel_df.iloc[0]['setpoint_h2']
                    fig.add_annotation(
                        x=x0_fuel, y=main_fuel_df.iloc[0]['exact_ocv'], xref="x", yref="y",
                        axref="pixel", ayref="pixel", text="BoT", showarrow=True, arrowhead=2, ax=-25, ay=-35, font=dict(color="#1f77b4", size=16),
                    )
                    fig.add_annotation(
                        x=x0_fuel, y=main_fuel_df.iloc[0]['leak_current'], xref="x", yref="y2",
                        axref="pixel", ayref="pixel", text="BoT", showarrow=True, arrowhead=2, ax=-25, ay=35, font=dict(color="#d62728", size=16),
                    )
                if not dashed_fuel_df.empty:
                    xN_fuel = dashed_fuel_df.iloc[-1]['setpoint_h2']
                    fig.add_annotation(
                        x=xN_fuel, y=dashed_fuel_df.iloc[-1]['exact_ocv'], xref="x", yref="y",
                        axref="pixel", ayref="pixel", text="EoT", showarrow=True, arrowhead=2, ax=30, ay=35, font=dict(color="#1f77b4", size=16),
                    )
                    fig.add_annotation(
                        x=xN_fuel, y=dashed_fuel_df.iloc[-1]['leak_current'], xref="x", yref="y2",
                        axref="pixel", ayref="pixel", text="EoT", showarrow=True, arrowhead=2, ax=30, ay=-35, font=dict(color="#d62728", size=16),
                    )
                plot_title = f"Fuel Flow Variation for {self.cell_id}" if view_mode == "Zoomed Out" else f"Fuel Flow Variation for {self.cell_id} (Zoomed In)"
                fig.update_layout(title=plot_title, template="plotly_white", width=1400, height=800)
                fig.update_xaxes(title_text="Brenngasfluss (in mL/min)")
                if view_mode == "Zoomed Out":
                    max_val = max(fuel_df['exact_ocv'].max(), fuel_df['leak_current'].max()) * 1.1
                    fig.update_yaxes(title_text="OCV [V]", secondary_y=False, range=[0, max_val], title_font=dict(color="#1f77b4"), tickfont=dict(color="#1f77b4"))
                    fig.update_yaxes(title_text="Leckstromdichte [A/cm²]", secondary_y=True, range=[0, max_val], title_font=dict(color="#d62728"), tickfont=dict(color="#d62728"))
                    self.figures[f"Fuel Flow Variation Plot - {self.cell_id}"] = fig
                else:
                    fig.update_yaxes(title_text="OCV [V]", secondary_y=False, rangemode="normal", title_font=dict(color="#1f77b4"), tickfont=dict(color="#1f77b4"))
                    fig.update_yaxes(title_text="Leckstromdichte [A/cm²]", secondary_y=True, rangemode="normal", title_font=dict(color="#d62728"), tickfont=dict(color="#d62728"))
                    self.figures[f"Fuel Flow Variation Plot - {self.cell_id} - Zoomed In"] = fig
            
            # Fuel Flow EIS Variation & Aging
            eis_fuel_title = f"Fuel Flow Variation of {self.cell_id} @{int(fuel_df.iloc[0]['exact_temp'])}°C and 100% H₂"
            fig_eis_fuel = self._create_base_figure(eis_fuel_title)
            for i, row in fuel_df.reset_index().iterrows():
                suffix = " (BoT)" if i == 0 else (" (EoT)" if i == len(fuel_df) - 1 else "")
                self._add_trace(fig_eis_fuel, row, f"{int(row['setpoint_h2'])} mL/min Fuel{suffix}", i)
            self.figures[eis_fuel_title] = fig_eis_fuel
            
            fuel_660 = fuel_df[fuel_df['setpoint_h2'] == 660]
            if len(fuel_660) >= 2:
                title = f"Aging of {self.cell_id} after Fuel Flow Variation"
                fig_aging_fuel = self._create_base_figure(title)
                self._add_trace(fig_aging_fuel, fuel_660.iloc[0], f"BoT (660 mL/min Fuel) OCV: {fuel_660.iloc[0]['exact_ocv']:.3f}V", 0)
                self._add_trace(fig_aging_fuel, fuel_660.iloc[-1], f"EoT (660 mL/min Fuel) OCV: {fuel_660.iloc[-1]['exact_ocv']:.3f}V", 1)
                self.figures[title] = fig_aging_fuel

         # --- NEW: H2/N2 COMPOSITION EIS PLOTS ---
        for phase_name, hum_label in [('H2/N2 Variation 0% H2O', '0%'), ('H2/N2 Variation 10% H2O', '10%')]:
            h2n2_df = df[df['phase'] == phase_name].sort_values('timestamp')
            
            if not h2n2_df.empty:
                title_h2n2 = f"H₂/N₂ Variation at {hum_label} H₂O for {self.cell_id} @ 650°C"
                
                if preview_data:
                    self._preview_table(h2n2_df, title_h2n2)
                    
                fig_h2n2 = self._create_base_figure(title_h2n2)
                
                for i, (_, row) in enumerate(h2n2_df.iterrows()):
                    # Extract the H2 flow and dynamically calculate the exact percentages 
                    # Assuming the total gas flow baseline is 660 mL/min (100%)
                    h2_val = row.get('setpoint_h2', 660)
                    
                    if pd.notna(h2_val):
                        h2_pct = int(round((h2_val / 660.0) * 100))
                        
                        if hum_label == '0%':
                            n2_pct = 100 - h2_pct
                            comp_label = f"{h2_pct}% H₂ / {n2_pct}% N₂"
                        else:
                            # If it's the 10% H2O plot, N2 takes up the remaining percentage
                            n2_pct = 100 - h2_pct - 10
                            comp_label = f"{h2_pct}% H₂ / {n2_pct}% N₂ / 10% H₂O"
                    else:
                        comp_label = "Unknown Composition"
                        
                    trace_name = f"{comp_label} (OCV: {row['exact_ocv']:.3f} V)"
                    self._add_trace(fig_h2n2, row, trace_name, i)
                
                self.figures[title_h2n2] = fig_h2n2

        # ----------------------------------------
        # --- 3. LEGACY HUMIDITY / TEMPERATURE / STABILITY PLOTS ---
        print("\n--- Unique Conditions Found in Data ---")
        if not df.empty:
            print(
                df[['temperature', 'humidity', 'flow_regime']]
                .drop_duplicates()
                .sort_values(by=['temperature', 'humidity'])
                .to_string(index=False)
            )
            
        # 1. EXTRACT ALL RELEVANT SCANS
        # Capture all 0% H2O, 650C scans. The first two correspond to Before/After Pol Curve.
        h2o_0_all = df[(df['humidity'] == 0) & (df['temperature'] == 650)].sort_values('timestamp')
        h2o_0 = h2o_0_all.head(1) # Preserve original h2o_0 definition for subsequent legacy plots
        
        h2o_30 = df[(df['humidity'] == 30) & (df['temperature'] == 650)].head(1)
        h2o_50_all = df[(df['humidity'] == 50) & (df['temperature'] == 650) & (df['flow_regime'] == 'High Flow')]
        h2o_50_initial = h2o_50_all.head(1)
        h2o_70 = df[(df['humidity'] == 70) & (df['temperature'] == 650)].head(1)
        h2o_90_high = df[(df['humidity'] == 90) & (df['temperature'] == 650) & (df['flow_regime'] == 'High Flow')].head(1)
        h2o_90_low = df[(df['humidity'] == 90) & (df['temperature'] == 650) & (df['flow_regime'] == 'Low Flow')].head(1)
        
        h2o_50_post_hum = pd.DataFrame()
        if not h2o_90_low.empty:
            h2o_50_post_hum = h2o_50_all[h2o_50_all['timestamp'] > h2o_90_low.iloc[0]['timestamp']].head(1)
        h2o_50_final = h2o_50_all.tail(1)
        
        t_750_start_df = df[df['temperature'] == 750]
        temp_var_df = pd.DataFrame()
        if not t_750_start_df.empty and not h2o_50_final.empty:
            t_750_start, t_final = t_750_start_df['timestamp'].min(), h2o_50_final.iloc[0]['timestamp']
            if pd.notna(t_750_start) and pd.notna(t_final):
                temp_var_df = df[(df['timestamp'] >= t_750_start) & (df['timestamp'] <= t_final)]
                temp_var_df = temp_var_df.sort_values(by=['temperature', 'humidity'], ascending=[False, True])
                temp_var_df = temp_var_df.drop_duplicates(subset=['temperature', 'humidity'], keep='first')

        # --- 2. NEW PLOT: EIS STABILITY CHECK AROUND POL CURVE ---
        if len(h2o_0_all) >= 2:
            title_pol = f"EIS Stability Check After FC Mode Pol Kurve for {self.cell_id}"
            pol_check_df = h2o_0_all.head(2)
            
            if preview_data:
                self._preview_table(pol_check_df, title_pol)
                
            fig_pol = self._create_base_figure(title_pol)
            eis_before = pol_check_df.iloc[0]
            eis_after = pol_check_df.iloc[1]
            
            self._add_trace(fig_pol, eis_before, f"Before Pol Curve (OCV: {eis_before['ocv']:.3f} V)", 0)
            self._add_trace(fig_pol, eis_after, f"After Pol Curve (OCV: {eis_after['ocv']:.3f} V)", 1)
            self.figures[title_pol] = fig_pol

        # --- 3. ORIGINAL PLOT: Baseline 0% H2O ---
        if not h2o_0.empty:
            row = h2o_0.iloc[0]
            title = f"EIS for {self.cell_id} ({self.test_name}) at 650°C (100% H₂) with an OCV of {row['ocv']:.3f} V"
            if preview_data:
                self._preview_table(h2o_0, title)
            fig = self._create_base_figure(title)
            self._add_trace(fig, row, f"100% H₂ / 0% H₂O (OCV: {row['ocv']:.3f} V)", 0)
            self.figures[title] = fig

        # Early Humidity Variation
        plot2_df = pd.concat([h2o_0, h2o_30, h2o_50_initial])
        if not plot2_df.empty:
            title = f"EIS Humidity Variation for {self.cell_id} ({self.test_name}) at 650°C (0%, 30% & 50%)"
            if preview_data:
                self._preview_table(plot2_df, title)
            fig = self._create_base_figure(title)
            for i, row in plot2_df.reset_index().iterrows():
                h_val = int(row['humidity']) if pd.notna(row['humidity']) else 0
                self._add_trace(fig, row, f"{h_val}% H₂O ({row['flow_regime']})", i)
            self.figures[title] = fig

        # Full Humidity Variation
        plot3_df = pd.concat([h2o_30, h2o_50_initial, h2o_70, h2o_90_high, h2o_90_low])
        if not plot3_df.empty:
            title = f"EIS Humidity Variation for {self.cell_id} ({self.test_name}) at 650°C"
            if preview_data:
                self._preview_table(plot3_df, title)
            fig = self._create_base_figure(title)
            for i, row in plot3_df.reset_index().iterrows():
                h_val = int(row['humidity']) if pd.notna(row['humidity']) else 0
                self._add_trace(fig, row, f"{h_val}% H₂O ({row['flow_regime']})", i)
            self.figures[title] = fig
            self._generate_eis_table(f"Resistances Table: Humidity Variation {self.cell_id}", plot3_df, "humidity")

        # Stability Check
        plot4_df = pd.concat([h2o_50_initial, h2o_50_post_hum])
        if len(plot4_df) == 2:
            title = f"Stability Check of {self.cell_id} ({self.test_name}) after High Humidity"
            if preview_data:
                self._preview_table(plot4_df, title)
            fig = self._create_base_figure(title)
            self._add_trace(fig, plot4_df.iloc[0], "Baseline 50% H₂O", 0)
            self._add_trace(fig, plot4_df.iloc[1], "Post-Hum 50% H₂O", 1)
            self.figures[title] = fig

        # Temperature Variation
        if not temp_var_df.empty:
            title = f"Temperature Variation for {self.cell_id} ({self.test_name})"
            if preview_data:
                self._preview_table(temp_var_df, title)
            fig = self._create_base_figure(title)
            for i, row in temp_var_df.reset_index().iterrows():
                t_val = int(row['temperature']) if pd.notna(row['temperature']) else 0
                h_val = int(row['humidity']) if pd.notna(row['humidity']) else 0
                self._add_trace(fig, row, f"{t_val}°C - {h_val}% H₂O ({row['flow_regime']})", i)
            self.figures[title] = fig
            self._generate_eis_table(f"Resistances Table: Temp Variation {self.cell_id}", temp_var_df, "temperature")

        # Aging Check
        plot6_df = pd.concat([h2o_50_initial, h2o_50_final])
        if len(plot6_df) == 2 and (h2o_50_initial.iloc[0]['timestamp'] != h2o_50_final.iloc[0]['timestamp']):
            title = f"Aging of {self.cell_id} ({self.test_name}) at 50% H₂O and 650°C"
            if preview_data:
                self._preview_table(plot6_df, title)
            fig = self._create_base_figure(title)
            self._add_trace(fig, h2o_50_initial.iloc[0], f"Initial (OCV: {h2o_50_initial.iloc[0]['ocv']:.3f} V)", 0)
            self._add_trace(fig, h2o_50_final.iloc[0], f"Final (OCV: {h2o_50_final.iloc[0]['ocv']:.3f} V)", 1)
            self.figures[title] = fig

        # --- 4. DISPLAY PLOTS ---
        print("\n--- Displaying Plots ---")
        if not self.figures:
            print("  -> No figures were generated.")
            return
            
        base_edits = {
            'annotationPosition': True, 'annotationTail': True, 'annotationText': True,
            'axisTitleText': True, 'titleText': True, 'colorbarPosition': True,
            'colorbarTitleText': True, 'legendPosition': True, 'legendText': True,
            'shapePosition': True,
        }
        
        for title, fig in self.figures.items():
            is_table = "Table" in title
            if not is_table:
                self._optimize_axes(fig, jc_mode=jc_mode)
                
                # Inject interactive "Delete Trace" dropdown to every plot before rendering
                dropdown_buttons = [dict(label="❌ Select Trace to Permanently Delete", method="skip")]
                for i, trace in enumerate(fig.data):
                    t_name = trace.name if trace.name else f"Trace {i}"
                    # Strip out html <b>, <sub> tags so it looks clean in the dropdown
                    clean_name = re.sub(r'<[^>]+>', '', str(t_name)) 
                    dropdown_buttons.append(
                        dict(
                            label=f"Delete: {clean_name[:45]}",
                            method="restyle",
                            args=[{"visible": False}, [i]]
                        )
                    )
                    
                existing_menus = list(fig.layout.updatemenus) if fig.layout.updatemenus else []
                existing_menus.append(
                    dict(
                        type="dropdown",
                        direction="up",
                        x=0.0, y=-0.15,
                        xanchor="left", yanchor="top",
                        showactive=False,
                        buttons=dropdown_buttons,
                        font=dict(size=12, color="red")
                    )
                )
                fig.update_layout(updatemenus=existing_menus)
            
            print(f"  -> Sending to browser: {title}")
            safe_filename = title.replace(')', '').replace('(', '').replace('°C', 'C').replace('/', '_').replace(' ', '_')
            
            if is_table:
                download_config = {'toImageButtonOptions': {'format': 'svg', 'filename': safe_filename, 'height': fig.layout.height, 'width': fig.layout.width, 'scale': 1}}
            elif jc_mode:
                download_config = {'toImageButtonOptions': {'format': 'png', 'filename': safe_filename, 'height': fig.layout.height, 'width': fig.layout.width, 'scale': 3}}
            else:
                download_config = {'toImageButtonOptions': {'format': 'svg', 'filename': safe_filename, 'height': fig.layout.height or 900, 'width': fig.layout.width or 1400, 'scale': 1}}
                
            download_config['editable'] = True
            download_config['edits'] = base_edits
            fig.show(config=download_config)

    # =========================================================================
    # 3. MATH & SCIENCE UTILITIES
    # =========================================================================
    
    def calculate_leak_current(self, T_C, OCV, vH2, vH2O, A, pO2_ox=0.21):
        """Calculates the Fuel Cell Leak Current natively based on exact Excel formulas."""
        if pd.isna(T_C) or not T_C:
            T_C = 650.0
        F = 96485.3321
        R = 8.314
        Vm = 22414
        T_K = T_C + 273.15
        nH2_t = vH2 / (Vm * 60)
        nH2O_t = vH2O / (Vm * 60)
        Kp = 10 ** (2.958 - (13022 / T_K))
        try:
            pO2_fuel = pO2_ox * math.exp((-4 * F * OCV) / (R * T_K))
        except OverflowError:
            pO2_fuel = 0
        ratio = math.sqrt(pO2_fuel) / Kp
        n_leak = (ratio * nH2_t - nH2O_t) / (2 * (1 + ratio))
        I_leak = (4 * F * n_leak) / A
        return I_leak

    def _get_crossings(self, raw_r, raw_i):
        crossings = []
        # Scan through adjacent points to find where the line crosses the x-axis (y = 0)
        for idx in range(len(raw_i) - 1):
            y1, y2 = raw_i[idx], raw_i[idx + 1]
            x1, x2 = raw_r[idx], raw_r[idx + 1]
            
            if (y1 <= 0 and y2 > 0) or (y1 > 0 and y2 <= 0):
                if y2 != y1:
                    # Linear interpolation to find the exact x-intercept
                    x_cross = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
                    crossings.append(x_cross)
                    
        # Sort the crossings from left to right (smallest Real value to largest)
        crossings = sorted(crossings)
        
        # --- 1. STANDARD CASES (Inductive Tail or Perfect Crossing) ---
        if len(crossings) >= 2:
            # Ideal case: crosses x-axis twice (High freq Rs, and Low freq Rtot)
            exact_rs = crossings[0]   
            exact_rtot = crossings[-1] 
            
        elif len(crossings) == 1:
            # Incomplete arc: crossed once at High Freq (Rs), test stopped early
            exact_rs = crossings[0]
            exact_rtot = max(raw_r)
            
        # --- 2. CAPACITIVE TAIL FIX (Hovering entirely above the axis) ---
        else:
            # Convert arrays to lists for safe indexing
            i_list = list(raw_i)
            r_list = list(raw_r)
            
            # Find the "Knee": The lowest vertical point (-Im(Z)) on the curve.
            # This perfectly identifies the bottom of the valley between the 
            # parasitic capacitive tail and the start of the true polarization arc.
            min_y_val = min(i_list)
            min_y_idx = i_list.index(min_y_val)
            
            exact_rs = r_list[min_y_idx]
            exact_rtot = max(raw_r)
            
        return exact_rs, exact_rtot

    # =========================================================================
    # 4. DATA PARSING HELPERS
    # =========================================================================
    def read_impedance_file(self, file_path):
        real_part, imag_part, frequencies = [], [], []
        try:
            with open(file_path, 'r', encoding='latin1') as file:
                lines = file.readlines()
            
            # --- AUTO-DETECT FORMAT ---
            is_polar = False
            data_start = 0
            
            # Scan headers to determine file type
            for i, line in enumerate(lines[:50]): # Check the first 50 lines for clues
                if 'Phase/deg' in line or 'Phase' in line:
                    is_polar = True
                if 'Impedance samples' in line:
                    is_polar = False
                    data_start = i + 4 # Old files start data exactly 4 lines after this string
                    break
            
            # --- PARSE DATA ---
            for j in range(data_start, len(lines)):
                line = lines[j]
                parts = line.strip().split()
                
                # Check if it looks like a data row (at least 4 columns)
                if len(parts) >= 4:
                    try:
                        # Safety check: ensure the first column is a number (like an index)
                        # If it fails, this is a header line and will skip to the 'except' block
                        _ = float(parts[0].replace(',', '.')) 
                        
                        freq = float(parts[1].replace(',', '.'))
                        val1 = float(parts[2].replace(',', '.'))
                        val2 = float(parts[3].replace(',', '.'))
                        
                        if is_polar:
                            # New Format (Polar): val1 = Magnitude, val2 = Phase
                            phase_rad = math.radians(val2)
                            r_val = val1 * math.cos(phase_rad)
                            i_val = val1 * math.sin(phase_rad)
                        else:
                            # Old Format (Rectangular): val1 = Real, val2 = Imaginary
                            r_val = val1
                            i_val = val2
                            
                        if r_val >= 0:
                            frequencies.append(freq)
                            real_part.append(r_val * self.cell_area)
                            # The electrochemistry standard is to plot -Im(Z) on the Y axis
                            imag_part.append(-i_val * self.cell_area)
                            
                    except ValueError:
                        # Skip headers, empty lines, or text comments
                        continue
                        
            if not frequencies:
                return None, None, None
                
            return np.array(real_part), np.array(imag_part), np.array(frequencies)
            
        except Exception as e:
            print(f"  -> Error reading {os.path.basename(file_path)}: {e}")
            return None, None, None

    # =========================================================================
    # 5. PLOTTING & UI HELPERS
    # =========================================================================
    
    def _preview_table(self, df_subset, plot_title):
        if df_subset.empty:
            return
        print(f"\n--- Data Preview for: {plot_title} ---")
        display_cols = ['filename', 'timestamp', 'temperature', 'humidity', 'flow_regime', 'ocv']
        preview_df = df_subset[display_cols].copy()
        preview_df['ocv'] = preview_df['ocv'].apply(lambda x: f"{x:.3f}")
        print(preview_df.to_string(index=False))

    def _generate_eis_table(self, title, plot_df, var_type):
        if var_type == 'humidity':
            headers = [
                ["<b>Humidity</b>", ""],
                ["<b>Flow Regime</b>", "<i> High / Low </i>"],
                ["<b>R<sub>s</sub></b>", "<i>(Ω·cm²)</i>"],
                ["<b>R<sub>pol</sub></b>", "<i>(Ω·cm²)</i>"],
            ]
        else:
            headers = [
                ["<b>Temperature</b>", ""],
                ["<b>Conditions</b>", "<i> Humidity & Flow </i>"],
                ["<b>R<sub>s</sub></b>", "<i>(Ω·cm²)</i>"],
                ["<b>R<sub>pol</sub></b>", "<i>(Ω·cm²)</i>"],
            ]
        row_data = [[], [], [], []]
        for _, row in plot_df.iterrows():
            rs, rtot = self._get_crossings(row['Re(Z)'], row['-Im(Z)'])
            rpol = (rtot - rs) if rs is not None and rtot is not None else None
            if var_type == 'humidity':
                row_data[0].append(f"{row['humidity']}%")
                row_data[1].append(row['flow_regime'])
            else:
                row_data[0].append(f"{row['temperature']}°C")
                row_data[1].append(f"{row['humidity']}% H₂O ({row['flow_regime']})")
            row_data[2].append(f"{rs:.4f}" if rs is not None else "-")
            row_data[3].append(f"{rpol:.4f}" if rpol is not None else "-")
        fig = go.Figure(
            data=[
                go.Table(
                    columnwidth=[0.8, 1.4, 1.0, 1.0],
                    header=dict(
                        values=headers, fill_color='#E1F0FA', font=dict(color='#212F3D', size=15, family="Arial"),
                        align='center', line_color='#D5D8DC', line_width=1, height=35,
                    ),
                    cells=dict(
                        values=row_data, fill_color='white', font=dict(color='#2C3E50', size=15, family="Arial"),
                        align='center', line_color='#D5D8DC', line_width=1, height=35,
                    ),
                )
            ]
        )
        calculated_height = 200 + (len(row_data[0]) * 35)
        fig.update_layout(
            title=dict(text=title, font=dict(family="Arial", size=18, color="black")),
            height=calculated_height, width=1000, margin=dict(t=60, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        self.figures[title] = fig

    def _create_base_figure(self, title):
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=title, font=dict(family="Arial", size=26, color="black")),
            template="plotly_white", height=900, width=1400, hovermode="closest",
            margin=dict(l=80, r=80, t=100, b=80),
            legend=dict(
                x=0.98, y=0.98, xanchor="right", yanchor="top", bordercolor="#888",
                borderwidth=1, bgcolor="rgba(255,255,255,0.9)", font=dict(family="Arial", size=16, color="#333"),
                itemclick="toggle", itemdoubleclick="toggleothers",
            ),
            xaxis=dict(title=dict(text="Re(Z) / Ω·cm²", font=dict(size=20)), tickfont=dict(size=18), showgrid=True, gridcolor="#E5E5E5", zerolinecolor="#888"),
            yaxis=dict(title=dict(text="-Im(Z) / Ω·cm²", font=dict(size=20)), tickfont=dict(size=18), showgrid=True, gridcolor="#E5E5E5", zerolinecolor="#888", scaleanchor="x", scaleratio=1),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=1.0, y=-0.15,
                    xanchor="right", yanchor="top",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Toggle Entire Legend",
                            method="relayout",
                            args=[{"showlegend": False}],
                            args2=[{"showlegend": True}],
                        )
                    ]
                )
            ]
        )
        return fig

    def _add_trace(self, fig, row, name, color_idx):
        import numpy as np # Ensure numpy is available for argmin
        color = self.colors[color_idx % len(self.colors)]
        re_arr, im_arr = row['Re(Z)'], row['-Im(Z)']
        freq_arr = row['Frequency']
        
        rs, rtot = self._get_crossings(re_arr, im_arr)
        rs_str = f"{rs:.4f}" if rs is not None else "-"
        rpol_str = f"{(rtot - rs):.4f}" if (rs is not None and rtot is not None) else "-"
        
        trace_name = f"{name} (R<sub>s</sub>: {rs_str}, R<sub>pol</sub>: {rpol_str})"
        
        # --- CAPACITIVE TAIL DETECTION AND DATA SLICING ---
        # Find the index of the "knee" (the lowest point on the y-axis)
        min_y_idx = np.argmin(im_arr)
        
        # Check if a severe capacitive tail exists:
        # 1. The knee is not the very first point.
        # 2. The start of the test (highest freq) is significantly higher on the y-axis than the knee.
        # 3. The start of the test is further left on the x-axis than the knee.
        if min_y_idx > 0 and im_arr[0] > (im_arr[min_y_idx] + 0.05) and re_arr[0] < re_arr[min_y_idx]:
            
            # Only keep the usable data (from the knee onwards)
            re_usable = re_arr[min_y_idx:]
            im_usable = im_arr[min_y_idx:]
            freq_usable = freq_arr[min_y_idx:]
            
            # Plot ONLY the True Fuel Cell Data (Bold & Marked)
            fig.add_trace(
                go.Scatter(
                    x=re_usable, y=im_usable, 
                    name=trace_name, 
                    mode='markers+lines',
                    marker=dict(size=8, color=color, symbol='circle'), 
                    line=dict(width=2.5, color=color),
                    hovertemplate=f"<b>{trace_name}</b><br>Re(Z): %{{x:.4f}} Ω·cm²<br>-Im(Z): %{{y:.4f}} Ω·cm²<br>Freq: %{{text:.1f}} Hz<extra></extra>",
                    text=freq_usable,
                )
            )
            
        else:
            # --- STANDARD PLOT (Healthy Inductive or Normal Data) ---
            fig.add_trace(
                go.Scatter(
                    x=re_arr, y=im_arr, name=trace_name, mode='markers+lines',
                    marker=dict(size=8, color=color, symbol='circle'), line=dict(width=2.5, color=color),
                    hovertemplate=f"<b>{trace_name}</b><br>Re(Z): %{{x:.4f}} Ω·cm²<br>-Im(Z): %{{y:.4f}} Ω·cm²<br>Freq: %{{text:.1f}} Hz<extra></extra>",
                    text=freq_arr,
                )
            )

    def _optimize_axes(self, fig, jc_mode=False):
        all_x, all_y = [], []
        if any(getattr(trace, 'yaxis', '') == 'y2' for trace in fig.data):
            return
        for trace in fig.data:
            if getattr(trace, 'x', None) is not None:
                all_x.extend(trace.x)
            if getattr(trace, 'y', None) is not None:
                all_y.extend(trace.y)
        if all_x and all_y:
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            x_range = max_x - min_x
            y_range = max_y - min_y
            if x_range == 0: x_range = 1.0
            if y_range == 0: y_range = 1.0
            pad_x = x_range * 0.1
            pad_y = y_range * 0.1
            y_lower_bound = min_y - pad_y
            if jc_mode:
                y_lower_bound = min(0, min_y - pad_y)
                actual_x_range = (max_x + pad_x) - (min_x - pad_x)
                actual_y_range = (max_y + pad_y) - y_lower_bound
                plot_width = 1400 - 80 - 80
                target_plot_height = plot_width * (actual_y_range / actual_x_range)
                target_figure_height = int(target_plot_height + 100 + 80)
                fig.update_layout(height=max(300, target_figure_height))
            fig.update_xaxes(range=[min_x - pad_x, max_x + pad_x])
            fig.update_yaxes(range=[y_lower_bound, max_y + pad_y])


# %%
def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    print("\n==============================================")
    print("  SOEC Data Analyzer & Plot Generator ")
    print("==============================================\n")
    print("Please choose a Color Theme for your plots:")
    
    COLOR_THEMES = {
        "1": {"name": "Default Vibrant (Plotly)", "colors": px.colors.qualitative.Plotly},
        "2": {"name": "Colorblind Safe (Safe)", "colors": px.colors.qualitative.Safe},
        "3": {"name": "Bold & Professional (Bold)", "colors": px.colors.qualitative.Bold},
        "4": {"name": "Pastel & Soft (Pastel)", "colors": px.colors.qualitative.Pastel},
        "5": {"name": "Scientific Vivid (Prism)", "colors": px.colors.qualitative.Prism}
    }
    
    for key, theme in COLOR_THEMES.items():
        print(f"  [{key}] - {theme['name']}")
        
    theme_choice = input("\nEnter theme number (1-5) [Press Enter for Default 1]: ").strip()
    selected_theme = COLOR_THEMES.get(theme_choice, COLOR_THEMES["1"])
    print(f"--> Selected Theme: {selected_theme['name']}\n")
    
    preview_choice = input("Would you like to preview the data tables used for each plot? (y/n) [Press Enter for 'n']: ").strip().lower()
    preview_data = preview_choice in ['y', 'yes']
    
    jc_mode_choice = input("Do you want to enter JC mode? (y/n) [Press Enter for 'n']: ").strip().lower()
    jc_mode = jc_mode_choice in ['y', 'yes']
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    folders = []
    is_multi = messagebox.askyesno("Multiple Folders?", "Is this test divided into more than one folder (e.g. Parts 1, 2, 3)?\n\nChoose 'Yes' to select multiple folders, or 'No' for a single folder.")
    
    if is_multi:
        print("\nPlease select the test folders one by one. Click 'Cancel' when you are done.")
        part_num = 1
        while True:
            f = filedialog.askdirectory(parent=root, title=f"Select Folder Part {part_num} (Click Cancel to finish)")
            if not f: break
            folders.append(f)
            part_num += 1
    else:
        print("\nPlease select the Test folder...")
        f = filedialog.askdirectory(parent=root, title="Select Test Folder")
        if f: folders.append(f)
        
    root.destroy()
    
    if folders:
        print(f"\nProcessing {len(folders)} Folder(s):")
        for f in folders: print(f" - {f}")
        
        config = {"custom_colors": selected_theme["colors"]}
        analyzer = SOECFootprintAnalyzer(folders=folders, config=config)
        analyzer.parse_log_file()
        analyzer.parse_eis_files()
        analyzer.generate_analyst_plots(preview_data=preview_data, jc_mode=jc_mode)
    else:
        print("No folder(s) selected. Exiting.")

if __name__ == '__main__':
    main()


