import os
import argparse
from typing import List, Optional, Dict, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# --- Configuration Parameters ---
DEFAULT_DATA_FILE_PATH = "TrioGT4o4.txt"
DEFAULT_OUTPUT_PATH = "preprocessed.csv"
DEFAULT_HIGHPASS_CUTOFF = 1.0
TIME_STEP_TOLERANCE = 1e-6
OUTLIER_ZSCORE_THRESHOLD = 3
MINIMUM_REQUIRED_LINES = 4
MINIMUM_REQUIRED_COLUMNS = 4
BUTTERWORTH_FILTER_ORDER = 4
SEGMENT_DURATION_SECONDS = 1.0
NORMALIZATION_MIN_VALUE = 0.0
NORMALIZATION_MAX_VALUE = 1.0
DECIMALS = 9
DEFAULT_SKIP_LINES = 3

# --- Helper Functions ---

def compute_sampling_rate(time_values: pd.Series) -> float:
    """Calculate the sampling rate from the time column values.

    Args:
        time_values (pd.Series): A pandas Series containing time stamps.

    Returns:
        float: The computed average sampling rate in Hz.
    """
    return 1 / np.mean(np.diff(time_values))

def check_uniform_time_step(time_values: pd.Series, tolerance: float = TIME_STEP_TOLERANCE) -> bool:
    """Check if time steps are approximately uniform within a given tolerance.

    Args:
        time_values (pd.Series): A pandas Series containing time stamps.
        tolerance (float): The maximum allowed deviation from the average time step.

    Returns:
        bool: True if time steps are uniform, False otherwise.
    """
    time_diffs = np.diff(time_values)
    avg_time_step = np.mean(time_diffs)
    max_deviation = np.max(np.abs(time_diffs - avg_time_step))
    if max_deviation > tolerance:
        print(f"Warning: Time steps are not uniform. Maximum deviation: {max_deviation:.6e}")
        return False
    return True

def replace_outliers_inplace(df: pd.DataFrame, column: str, threshold: float = OUTLIER_ZSCORE_THRESHOLD) -> None:
    """Replace outliers in a DataFrame column with interpolated values, modifying it in place.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        column (str): The name of the column to clean.
        threshold (float): The Z-score threshold to identify outliers.
    """
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    outliers = np.abs(z_scores) > threshold
    if outliers.any():
        print(f"Replacing {outliers.sum()} outliers in column '{column}'.")
        df[column] = df[column].mask(outliers).interpolate(method='linear', limit_direction='both')

def butterworth_highpass_filter(data: np.ndarray, cutoff_freq: float, sampling_rate: float, order: int) -> np.ndarray:
    """Apply a Butterworth highpass filter to the input signal.

    Args:
        data (np.ndarray): The input signal array.
        cutoff_freq (float): The cutoff frequency of the filter in Hz.
        sampling_rate (float): The sampling rate of the signal in Hz.
        order (int): The order of the Butterworth filter.

    Returns:
        np.ndarray: The filtered signal.
    """
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

# --- Normalization Functions ---

def normalize_segmented(signal: pd.Series, sampling_rate: float) -> np.ndarray:
    """Normalize signal in fixed-duration segments.

    Args:
        signal (pd.Series): The signal to normalize.
        sampling_rate (float): The sampling rate of the signal.

    Returns:
        np.ndarray: The normalized signal.
    """
    samples_per_segment = int(SEGMENT_DURATION_SECONDS * sampling_rate)
    normalized_signal = []
    
    for start_idx in range(0, len(signal), samples_per_segment):
        segment = signal.iloc[start_idx : start_idx + samples_per_segment]
        if not segment.empty:
            segment_min, segment_max = segment.min(), segment.max()
            range_val = segment_max - segment_min
            if range_val == 0:
                normalized_segment = np.full_like(segment, (NORMALIZATION_MIN_VALUE + NORMALIZATION_MAX_VALUE) / 2)
            else:
                normalized_segment = NORMALIZATION_MIN_VALUE + (segment - segment_min) * (NORMALIZATION_MAX_VALUE - NORMALIZATION_MIN_VALUE) / range_val
            normalized_signal.extend(np.clip(normalized_segment, NORMALIZATION_MIN_VALUE, NORMALIZATION_MAX_VALUE))
            
    return np.array(normalized_signal)

def normalize_rolling(signal: pd.Series, sampling_rate: float) -> np.ndarray:
    """Normalize signal using a rolling window to avoid discontinuities.

    Args:
        signal (pd.Series): The signal to normalize.
        sampling_rate (float): The sampling rate of the signal.

    Returns:
        np.ndarray: The normalized signal.
    """
    window_size = int(SEGMENT_DURATION_SECONDS * sampling_rate)
    rolling_min = signal.rolling(window=window_size, min_periods=1, center=True).min()
    rolling_max = signal.rolling(window=window_size, min_periods=1, center=True).max()
    
    signal_range = rolling_max - rolling_min
    signal_range.replace(0, 1.0, inplace=True) # Avoid division by zero
    
    normalized_signal = NORMALIZATION_MIN_VALUE + (signal - rolling_min) * (NORMALIZATION_MAX_VALUE - NORMALIZATION_MIN_VALUE) / signal_range
    return np.clip(normalized_signal.to_numpy(), NORMALIZATION_MIN_VALUE, NORMALIZATION_MAX_VALUE)

# --- Normalization Factory ---
NORMALIZATION_DISPATCHER: Dict[str, Callable[[pd.Series, float], np.ndarray]] = {
    'segment': normalize_segmented,
    'rolling': normalize_rolling,
}

# --- Core Functions ---

def load_and_validate_data(file_path: str, skip_lines: int) -> Optional[pd.DataFrame]:
    """Load and validate data from the input file using pandas for efficiency.

    Args:
        file_path (str): The path to the input data file.
        skip_lines (int): The number of header lines to skip.

    Returns:
        Optional[pd.DataFrame]: A DataFrame with the loaded data, or None if validation fails.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return None

    try:
        df = pd.read_csv(
            file_path,
            skiprows=skip_lines,
            sep=',',  # Changed to comma delimiter
            header=None,
            usecols=[0, 1, 2, 3],
            names=["time", "APW", "ECG", "PPG"],
            on_bad_lines='warn'
        )

        # Print structure for inspection
        print("\n--- DataFrame Structure ---")
        print(f"Columns: {df.columns.tolist()}")
        print("First 5 rows:")
        print(df.head())
        print(f"Shape: {df.shape}\n")

        if len(df) < MINIMUM_REQUIRED_LINES:
            print(f"Error: File has fewer than {MINIMUM_REQUIRED_LINES} data lines.")
            return None
        
        for col in ["APW", "ECG", "PPG"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        df['time'] = df['time'].astype(float).round(DECIMALS)
        df = df.drop_duplicates(subset=["time"]).reset_index(drop=True)
        return df

    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None

def process_signals(df: pd.DataFrame, highpass_cutoff: float, normalization_method: str) -> Optional[pd.DataFrame]:
    """Run the main preprocessing steps on the loaded data.

    Args:
        df (pd.DataFrame): The input DataFrame with signal data.
        highpass_cutoff (float): The high-pass filter cutoff frequency in Hz.
        normalization_method (str): The name of the normalization method to use.

    Returns:
        Optional[pd.DataFrame]: The processed DataFrame, or None if a critical error occurs.
    """
    print("Starting signal processing...")
    
    if not check_uniform_time_step(df["time"]):
        print("Error: Non-uniform time step detected. Aborting processing.")
        return None
        
    for col in [ "APW", "ECG", "PPG"]:
        replace_outliers_inplace(df, col)
        
    sampling_rate = compute_sampling_rate(df["time"])
    print(f"Computed Sampling Rate: {sampling_rate:.2f} Hz")
    
    print(f"Applying High-pass filter with cutoff: {highpass_cutoff} Hz")
    df['highpass_APW'] = butterworth_highpass_filter(
        df["APW"].values, highpass_cutoff, sampling_rate, BUTTERWORTH_FILTER_ORDER
    )
    
    print(f"Applying '{normalization_method}' normalization...")
    signal_to_normalize = df['highpass_APW']
    
    norm_func = NORMALIZATION_DISPATCHER.get(normalization_method)
    if norm_func:
        df["normalized_APW"] = norm_func(signal_to_normalize, sampling_rate)
    else:
        print(f"Error: Unknown normalization method '{normalization_method}'.")
        return None
        
    print("Processing complete.")
    return df

def generate_and_save_plots(df: pd.DataFrame, save_plots: bool, output_prefix: str = "plot") -> None:
    """Generate plots and either show or save them to files.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        save_plots (bool): If True, save plots to files. Otherwise, display them.
        output_prefix (str): The prefix for saved plot filenames.
    """
    print("Generating plots...")
    
    plt.figure(figsize=(12, 6))
    plt.plot(df["time"], df['highpass_APW'], label="High-pass Filtered APW")
    plt.title("High-pass Filtered APW Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    if save_plots:
        filename = f"{output_prefix}_highpass.png"
        plt.savefig(filename)
        print(f"Saved high-pass plot to {filename}")
    else:
        plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(df["time"], df["normalized_APW"], label="Normalized APW")
    plt.title("Normalized APW Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude (Normalized)")
    plt.legend()
    plt.grid(True)

    if save_plots:
        filename = f"{output_prefix}_normalized.png"
        plt.savefig(filename)
        print(f"Saved normalization plot to {filename}")
    else:
        plt.show()
    plt.close()

# --- Main Execution ---

def main() -> None:
    """Main function to parse arguments and orchestrate the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess time-series data from a text file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_file", type=str, default=DEFAULT_DATA_FILE_PATH, help="Path to the input data file.")
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_PATH, help="Path to save the preprocessed data file.")
    parser.add_argument("--highpass_cutoff", type=float, default=DEFAULT_HIGHPASS_CUTOFF, help="High-pass filter cutoff frequency in Hz.")
    parser.add_argument("--skip_lines", type=int, default=DEFAULT_SKIP_LINES, help="Number of header lines to skip in the input file.")
    parser.add_argument("--normalization_method", type=str, choices=NORMALIZATION_DISPATCHER.keys(), default='segment', help="Method for signal normalization.")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to files instead of displaying them interactively.")
    
    args = parser.parse_args()

    # 1. Load and Validate
    df = load_and_validate_data(args.input_file, args.skip_lines)
    if df is None:
        print("Halting due to data loading failure.")
        return

    # 2. Process Signals
    processed_df = process_signals(df, args.highpass_cutoff, args.normalization_method)
    if processed_df is None:
        print("Halting due to data processing failure.")
        return

    # 3. Generate Plots
    generate_and_save_plots(processed_df, args.save_plots)

    # 4. Save Data
    processed_df.to_csv(args.output_file, index=False)
    print(f"Preprocessed data successfully saved to {args.output_file}.")

if __name__ == "__main__":
    main()