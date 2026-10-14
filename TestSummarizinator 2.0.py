# %%
import os
import glob
import re
import datetime
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

# %%
class LogParser:
    def __init__(self, folders):
        self.folders = folders
        self.cell_id = "Unknown_Cell"
        self.timeline = []
        self.df_files = pd.DataFrame()

    def parse_log_files(self):
        print("\n--- 1. Parsing Master Log(s) ---")
        
        timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})|(\d{2}\.\d{2}\.\d{4}\s\d{2}:\d{2}:\d{2})')
        cell_name_pattern = re.compile(r'Cell-Name\s*:\s*([A-Za-z0-9_-]+)', re.IGNORECASE)
        exact_h2_pattern = re.compile(r'Anode-Wasserstoff \(*H2\)*:\s*([\d,.]+)\s*ml/min\s*([\d,.]+)\s*ml/min', re.IGNORECASE)
        exact_o2_pattern = re.compile(r'Anode-Sauerstoff \(*O2\)*:\s*([\d,.]+)\s*ml/min\s*([\d,.]+)\s*ml/min', re.IGNORECASE)
        exact_air_pattern = re.compile(r'Kathode-Luft:\s*([\d,.]+)\s*(?:ml/min)?\s*([\d,.]+)\s*(?:ml/min)?', re.IGNORECASE)
        
        temp_pattern = re.compile(r'set T901000 temperature to ([\d.,]+)|Target Temperature:\s*([\d.,]+)|Temperature:\s*([\d.,]+)', re.IGNORECASE)
        hum_pattern = re.compile(r'pH2O\s*=\s*(\d+)%', re.IGNORECASE)
        phase_pattern = re.compile(r'Next step:\s*(.*)', re.IGNORECASE) # Captures your VBScript log strings
        
        for folder in self.folders:
            log_files = (glob.glob(os.path.join(folder, "*LogFile*.txt")) + glob.glob(os.path.join(folder, "*.log")))
            if not log_files:
                continue
                
            log_path = log_files[0]
            print(f"  -> Found log file: {os.path.basename(log_path)}")
            
            try:
                with open(log_path, 'r', encoding='latin1') as f:
                    cell_match = cell_name_pattern.search(f.read())
                    if cell_match and self.cell_id == "Unknown_Cell": self.cell_id = cell_match.group(1).strip()
            except Exception: pass
            
            current_temp, current_humidity = 650, 0
            current_phase = "Initial Reduction / Base"
            in_param_info = False
            setpoint_h2, setpoint_o2, setpoint_air = 660.0, 0.0, 1400.0
            
            with open(log_path, 'r', encoding='latin1') as f:
                for line in f:
                    if "Sub Parameter_Info, START" in line: in_param_info = True
                    elif "Sub Parameter_Info, END" in line: in_param_info = False
                    
                    if in_param_info:
                        h2_match = exact_h2_pattern.search(line)
                        if h2_match: setpoint_h2 = float(h2_match.group(2).replace(',', '.'))
                            
                        o2_match = exact_o2_pattern.search(line)
                        if o2_match: setpoint_o2 = float(o2_match.group(2).replace(',', '.'))
                        
                        air_match = exact_air_pattern.search(line)
                        if air_match: setpoint_air = float(air_match.group(2).replace(',', '.'))
                    
                    temp_match = temp_pattern.search(line)
                    if temp_match: 
                        raw_temp = float(next(g for g in temp_match.groups() if g is not None).replace(',', '.'))
                        current_temp = int(round(raw_temp / 5.0) * 5.0) 
                    
                    hum_match = hum_pattern.search(line)
                    if hum_match: current_humidity = int(hum_match.group(1))
                    
                    # Track exactly what phase we are in from the script
                    phase_match = phase_pattern.search(line)
                    if phase_match: current_phase = phase_match.group(1).strip()

                    ts_match = timestamp_pattern.search(line)
                    if ts_match:
                        ts_str = ts_match.group(1) or ts_match.group(2)
                        dt = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S") if '-' in ts_str else datetime.datetime.strptime(ts_str, "%d.%m.%Y %H:%M:%S")
                        
                        self.timeline.append({
                            'timestamp': dt, 
                            'phase': current_phase,
                            'temperature': current_temp, 
                            'humidity': current_humidity, 
                            'setpoint_h2': setpoint_h2,
                            'setpoint_o2': setpoint_o2,
                            'setpoint_air': setpoint_air
                        })
                        
        self.timeline.sort(key=lambda x: x['timestamp'])

    def parse_data_files(self):
        print("\n--- 2. Mapping Data Files to Unified Timeline ---")
        if not self.timeline: return
        
        all_files = []
        for folder in self.folders:
            for sub_dir in ["Data/EIS_orig", "Data/Pol_orig", "EIS_orig", "Pol_orig"]:
                target = os.path.join(folder, sub_dir.replace("/", os.sep))
                if os.path.exists(target): all_files.extend(glob.glob(os.path.join(target, "*.txt")))
        
        mapped_data = []
        for file_path in all_files:
            filename = os.path.basename(file_path)
            match = re.search(r'(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', filename)
            if not match: continue
            
            yy, mm, dd, hr, mn, sc = match.groups()
            file_time = datetime.datetime(int("20"+yy), int(mm), int(dd), int(hr), int(mn), int(sc))
            
            current_state = self.timeline[0].copy()
            for event in self.timeline:
                if event['timestamp'] <= file_time: current_state.update(event)
                else: break
            
            mapped_data.append({'filename': filename, 'file_time': file_time, **current_state})
                
        if mapped_data:
            self.df_files = pd.DataFrame(mapped_data).sort_values('file_time')

    def _get_big_steps_df(self):
        if self.df_files.empty: return pd.DataFrame()
            
        df = self.df_files.copy()
        
        # Trigger a new block if ANY condition or flow rate changes!
        df['condition_changed'] = (
            (df['temperature'] != df['temperature'].shift()) | 
            (df['humidity'] != df['humidity'].shift()) | 
            (df['setpoint_h2'] != df['setpoint_h2'].shift()) |
            (df['setpoint_o2'] != df['setpoint_o2'].shift()) |
            (df['setpoint_air'] != df['setpoint_air'].shift()) |
            (df['phase'] != df['phase'].shift()) # Also trigger if you print a new "Next step"
        )
        df['is_new_eis'] = df['filename'].str.lower().str.startswith('eis_')
        df['new_block'] = df['condition_changed'] | df['is_new_eis']
        df['block_id'] = df['new_block'].cumsum()
        
        big_steps = []
        for block_id, group in df.groupby('block_id'):
            filenames = group['filename'].dropna().unique()
            filtered_names = [f for f in filenames if not f.upper().startswith('OCV_')]
            if len(filtered_names) == 0: continue
                
            eis_files = [f for f in filtered_names if f.lower().startswith('eis_')]
            sofc_files = [f for f in filtered_names if 'sofc' in f.lower()]
            soec_files = [f for f in filtered_names if 'soec' in f.lower()]
            
            first_row = group.iloc[0]
            
            h2_flow = int(round(first_row.get('setpoint_h2', 0)))
            o2_flow = int(round(first_row.get('setpoint_o2', 0)))
            air_flow = int(round(first_row.get('setpoint_air', 0)))
            
            # Mathematically calculate N2 (Assuming 660 total anode flow)
            n2_flow = max(0, int(660 - h2_flow - (o2_flow * 2)))
            
            big_steps.append({
                'Test Phase (from Log)': first_row.get('phase', 'Unknown Phase'),
                'Temp': int(first_row['temperature']), 
                'Air Flow': air_flow,
                'H₂ Flow': h2_flow,
                'N₂ Flow': n2_flow,
                'O₂ Flow': o2_flow,
                'Hum %': int(first_row['humidity']),
                'EIS data': "\n".join(eis_files) if eis_files else "-",
                'IV SOFC': "\n".join(sofc_files) if sofc_files else "-",
                'IV SOEC': "\n".join(soec_files) if soec_files else "-",
                'Timestamp': first_row['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                'raw_phase_color': str(first_row.get('phase', '')).lower()
            })
            
        df_big = pd.DataFrame(big_steps)
        
        # Clean up repeated Phase names for a beautiful display
        cleaned_phases = []
        last_phase = None
        for phase in df_big['Test Phase (from Log)']:
            if phase != last_phase:
                cleaned_phases.append(phase)
                last_phase = phase
            else:
                cleaned_phases.append("")
        
        df_big['Test Phase (from Log)'] = cleaned_phases
        return df_big

    def generate_data_previews(self):
        print("\n--- Data Preview: Unique Big Steps ---")
        df_big = self._get_big_steps_df()
        if df_big.empty:
            print("  -> No data available.")
            return
        
        df_console = df_big.drop(columns=['raw_phase_color']).copy()
        for col in ['EIS data', 'IV SOFC', 'IV SOEC']:
            df_console[col] = df_console[col].str.replace("\n", " ")
        print(df_console.to_string(index=False))

    def export_to_excel(self, output_filename="Test_Big_Steps_Summary.xlsx"):
        print("\n--- 3. Exporting Styled Big Steps to Excel ---")
        df_export = self._get_big_steps_df()
        if df_export.empty: return
            
        output_path = os.path.join(self.folders[0], output_filename)
        try:
            with open(output_path, 'a'): pass
        except PermissionError:
            output_filename = output_filename.replace(".xlsx", "_v2.xlsx")
            output_path = os.path.join(self.folders[0], output_filename)
        except Exception: pass
            
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                start_row_excel = 8  
                start_col_excel = 4  
                
                # Drop the hidden color helper column before exporting
                df_to_write = df_export.drop(columns=['raw_phase_color'])
                df_to_write.to_excel(writer, index=False, startrow=start_row_excel - 1, startcol=start_col_excel - 1, sheet_name='Summary')
                
                worksheet = writer.sheets['Summary']
                from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
                from openpyxl.utils import get_column_letter
                
                title_font = Font(name='Calibri', size=16, bold=True, color="FFFFFF")
                title_fill = PatternFill(start_color="5B9BD5", end_color="5B9BD5", fill_type="solid")
                header_font = Font(name='Calibri', size=11, bold=True, color="FFFFFF")
                header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
                thin_border = Border(left=Side(style='thin', color='BFBFBF'), right=Side(style='thin', color='BFBFBF'), top=Side(style='thin', color='BFBFBF'), bottom=Side(style='thin', color='BFBFBF'))
                
                def get_row_color(phase_str):
                    """Dynamically colors the row based on keywords found in the user's VBScript log."""
                    p = phase_str.lower()
                    if 'h2/n2' in p and '10%' in p: return "FCE4D6" # Light Orange
                    if 'h2/n2' in p: return "FFF2CC"                # Light Yellow
                    if 'airflow' in p or 'air flow' in p: return "E2EFDA" # Light Green
                    if 'fuelflow' in p or 'fuel flow' in p: return "DDEBF7" # Light Blue
                    if 'humid' in p or 'h2o' in p: return "D9E1F2"  # Darker Blue
                    if 'temp' in p: return "F4B084"                 # Solid Orange
                    if 'base' in p or 'reduction' in p: return "F2F2F2" # Light Gray
                    return "FFFFFF" # White
                
                end_col_excel = start_col_excel + len(df_to_write.columns) - 1
                
                # Title Banner
                title_row = start_row_excel - 2
                worksheet.merge_cells(start_row=title_row, start_column=start_col_excel, end_row=title_row, end_column=end_col_excel)
                title_cell = worksheet.cell(row=title_row, column=start_col_excel)
                title_cell.value = f"Dynamic Test Sequence Summary - {self.cell_id}"
                title_cell.font = title_font
                title_cell.fill = title_fill
                title_cell.alignment = center_align
                worksheet.row_dimensions[title_row].height = 28
                
                # Format Table Headers
                for col_num in range(start_col_excel, end_col_excel + 1):
                    cell = worksheet.cell(row=start_row_excel, column=col_num)
                    cell.font = header_font
                    cell.fill = header_fill
                    cell.alignment = center_align
                    cell.border = thin_border
                worksheet.row_dimensions[start_row_excel].height = 20
                
                # Format Data Cells using Dynamic Colors
                for row_idx, row_num in enumerate(range(start_row_excel + 1, start_row_excel + 1 + len(df_to_write))):
                    raw_phase = df_export.iloc[row_idx]['raw_phase_color']
                    hex_color = get_row_color(raw_phase)
                    row_fill = PatternFill(start_color=hex_color, end_color=hex_color, fill_type="solid")
                    
                    for col_idx, col_num in enumerate(range(start_col_excel, end_col_excel + 1)):
                        cell = worksheet.cell(row=row_num, column=col_num)
                        cell.alignment = center_align
                        cell.border = thin_border
                        cell.fill = row_fill
                        
                        col_name = df_to_write.columns[col_idx]
                        if col_name == 'Test Phase (from Log)': cell.font = Font(name='Calibri', size=11, bold=True, italic=True)
                        elif col_name in ['EIS data', 'IV SOFC', 'IV SOEC']: cell.font = Font(name='Calibri', size=11, italic=True)
                        else: cell.font = Font(name='Calibri', size=11)
                
                # Auto-Adjust Column Widths
                for col_num in range(start_col_excel, end_col_excel + 1):
                    col_letter = get_column_letter(col_num)
                    max_length = len(str(worksheet.cell(row=start_row_excel, column=col_num).value))
                    
                    for row_num in range(start_row_excel + 1, start_row_excel + 1 + len(df_to_write)):
                        cell_val = str(worksheet.cell(row=row_num, column=col_num).value)
                        if cell_val != 'None' and cell_val != '-' and cell_val != '':
                            lines = cell_val.split('\n')
                            longest = max([len(line) for line in lines])
                            if longest > max_length: max_length = longest
                                
                    worksheet.column_dimensions[col_letter].width = min(max_length + 3, 35) 
            print(f"  -> Successfully generated aesthetic Excel file: {output_path}")
        except Exception as e:
            print(f"  -> ERROR writing Excel file: {e}")

# %%
def main():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    folders = []
    if messagebox.askyesno("Multiple Folders?", "Is this test divided into more than one folder?\nChoose 'Yes' to select multiple, or 'No' for a single folder."):
        part_num = 1
        while True:
            f = filedialog.askdirectory(title=f"Select Folder Part {part_num} (Click Cancel to finish)")
            if not f: break
            folders.append(f)
            part_num += 1
    else:
        f = filedialog.askdirectory(title="Select Test Folder")
        if f: folders.append(f)
            
    if not folders: return
        
    print(f"\nProcessing {len(folders)} Folder(s)...")
    parser = LogParser(folders=folders)
    parser.parse_log_files()
    parser.parse_data_files()
    parser.generate_data_previews()
    parser.export_to_excel(output_filename="Test_Big_Steps_Summary.xlsx")

if __name__ == "__main__":
    main()