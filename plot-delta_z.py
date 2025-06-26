import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# ===================================================================
# === 1. Default Configuration
# ===================================================================
CONFIG = {
    "lambda_": 1.5513e-06,
    "n": 1.0003,
    "z_offset": 0,
    "input_file": "time_signal_status-demo.csv",
    "resampled_file": "resampled.csv",
    "filter": {
        "order": 8,
        "cutoff_hz": 0.5,
        "sampling_rate_hz": 100,
    },
    "plotting": {
        "font_size": 20,
        "figure_size": (13, 5),
    }
}

# ===================================================================
# === 2. Core Functions (plot_results is slightly modified)
# ===================================================================

def setup_plot_style(font_size):
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size, labelsize=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('figure', titlesize=font_size)

def load_and_validate_data(file_path, required_columns):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Input file '{file_path}' not found.")
    df = pd.read_csv(file_path)
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise KeyError(f"Missing required columns: {missing}")
    return df

def calculate_displacement(df, lambda_, n, z_offset):
    hwn_list, dz_list = [], []
    direction, half_wave_number = -1, 0.0
    first_bp_reached, change_direction_flag = False, False

    for row in df.itertuples(index=False):
        if row.Status == "IGNORE":
            hwn_list.append(half_wave_number)
            dz_list.append((half_wave_number * lambda_) / (4 * n) - z_offset)
            continue
        if row.Status == "BP":
            first_bp_reached, change_direction_flag = True, True
        elif first_bp_reached and change_direction_flag:
            direction *= -1
            change_direction_flag = False
        current_hwn = half_wave_number
        if row.Status != "BP" and first_bp_reached:
            current_hwn += direction
        hwn_to_log = current_hwn - 0.5 if row.Status == "BP" else current_hwn
        hwn_list.append(hwn_to_log)
        dz_list.append((hwn_to_log * lambda_) / (4 * n) - z_offset)
        half_wave_number = current_hwn
    df_out = df.copy()
    df_out["half_wave_number"], df_out["delta_z"] = hwn_list, dz_list
    return df_out

def interpolate_displacement(df):
    df_sorted = df.sort_values(by="Time").drop_duplicates(subset="Time", keep="first")
    mask_minmax = df_sorted["Status"].isin(["MIN", "MAX"])
    time_minmax, z_minmax = df_sorted.loc[mask_minmax, "Time"].values, df_sorted.loc[mask_minmax, "delta_z"].values
    if len(time_minmax) < 2:
        print("Warning: Not enough MIN/MAX points (<2) for interpolation.")
        return df_sorted["Time"].values, df_sorted["delta_z"].values
    interp_func = interp1d(time_minmax, z_minmax, kind='cubic', fill_value="extrapolate")
    time_dense = np.linspace(time_minmax.min(), time_minmax.max(), 5000)
    return time_dense, interp_func(time_dense)

# --- MODIFIED plot_results function ---
def plot_results(time, delta_z, plot_cfg, title, output_path=None, show_plot=True): # <-- NEW arguments
    """Generates, saves, and/or shows the final plot."""
    fig = plt.figure(figsize=plot_cfg["figure_size"])
    plt.plot(time, delta_z * 1e6, label=r"$\Delta z$ Displacement", color="red", linewidth=2)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\Delta z$ [μm]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if output_path: # <-- NEW: Save the figure if path is provided
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")

    if show_plot: # <-- NEW: Show the plot based on the flag
        plt.show()
    
    plt.close(fig) # <-- NEW: Close the figure to free up memory

# ===================================================================
# === 3. Main Execution Block
# ===================================================================
def main():
    """Main script execution."""
    parser = argparse.ArgumentParser(
        description="Process Fabry-Perot interferometer data to calculate displacement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # <-- Shows defaults in help
    )
    
    # --- Group for File Arguments ---
    file_group = parser.add_argument_group('File I/O')
    file_group.add_argument('--input-file', type=str, default=CONFIG["input_file"], help="Path to the primary input CSV file.")
    file_group.add_argument('--resampled-file', type=str, default=CONFIG["resampled_file"], help="Path to the resampled ChannelC CSV file.") # <-- NEW
    file_group.add_argument('--output-csv', type=str, default=None, help="Path to save the processed DataFrame to a CSV file.") # <-- NEW
    file_group.add_argument('--output-plot', type=str, default=None, help="Path to save the output plot image (e.g., plot.png).") # <-- NEW

    # --- Group for Physics Parameters ---
    phys_group = parser.add_argument_group('Physics Parameters')
    phys_group.add_argument('--wavelength', type=float, default=CONFIG["lambda_"], help="Wavelength (lambda) in meters.") # <-- NEW
    phys_group.add_argument('--refractive-index', type=float, default=CONFIG["n"], help="Refractive index (n).") # <-- NEW
    phys_group.add_argument('--z-offset', type=float, default=CONFIG["z_offset"], help="Z-offset in meters.") # <-- NEW

    # --- Group for Filter Parameters ---
    filt_group = parser.add_argument_group('Filter Parameters')
    filt_group.add_argument('--filter-order', type=int, default=CONFIG["filter"]["order"], help="Order of the low-pass filter.") # <-- NEW
    filt_group.add_argument('--cutoff-freq', type=float, default=CONFIG["filter"]["cutoff_hz"], help="Cutoff frequency of the low-pass filter in Hz.") # <-- NEW
    
    # --- Group for Execution Control ---
    exec_group = parser.add_argument_group('Execution Control')
    exec_group.add_argument('--no-plot', action='store_true', help="Do not display the plot interactively.") # <-- NEW
    exec_group.add_argument('--quiet', action='store_true', help="Suppress non-error print messages.") # <-- NEW

    args = parser.parse_args()
    
    # --- Update CONFIG from command-line arguments ---
    cfg = CONFIG.copy() # Start with defaults, then override
    cfg["input_file"] = args.input_file
    cfg["resampled_file"] = args.resampled_file
    cfg["lambda_"] = args.wavelength
    cfg["n"] = args.refractive_index
    cfg["z_offset"] = args.z_offset
    cfg["filter"]["order"] = args.filter_order
    cfg["filter"]["cutoff_hz"] = args.cutoff_freq
    
    if not args.quiet:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"--- Running analysis on: {cfg['input_file']} ---")

    setup_plot_style(cfg["plotting"]["font_size"])
    df_raw = load_and_validate_data(cfg["input_file"], {"Time", "Status", "Signal"})
    
    df_processed = calculate_displacement(df_raw, cfg["lambda_"], cfg["n"], cfg["z_offset"])
    
    if not args.quiet:
        print("\nProcessed DataFrame (First 20 Rows):")
        pd.set_option('display.float_format', '{:.9f}'.format)
        print(df_processed.head(20))
    
    if args.output_csv: # <-- NEW: Save the processed data if requested
        df_processed.to_csv(args.output_csv, index=False)
        if not args.quiet:
            print(f"\nProcessed data saved to {args.output_csv}")
            
    time_interp, delta_z_interp = interpolate_displacement(df_processed)
    
    plot_results(
        time_interp, delta_z_interp, cfg["plotting"], 
        r"Time vs. Interpolated $\Delta z$ Displacement",
        output_path=args.output_plot, # Pass the output path
        show_plot=not args.no_plot   # Show plot unless --no-plot is used
    )

if __name__ == "__main__":
    main()