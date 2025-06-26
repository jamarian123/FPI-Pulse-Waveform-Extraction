# === Configurable Hardcoded Values (set by argparse below) ===
MIN_DISTANCE_POINTS = None  # Minimum distance in data points between consecutive extrema
PEAK_HEIGHT_THRESHOLD = None  # Threshold for peak detection
PROMINENCE = None  # Prominence for peak and minima detection
START_TIME_S = None  # Start time in seconds
END_TIME_S = None  # End time in seconds
VIEWING_WINDOW_S = None  # The length of the viewing window in seconds

# Constants for Breaking Point (BP) detection logic
BP_TIME_WINDOW = None  # Time window (in seconds) around an absolute time line to look for a BP
BP_SIGNAL_RANGE_STRICT = None  # Strict signal amplitude range for a point to be a BP

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import os
import argparse
from matplotlib.widgets import Slider
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List, Optional

class SignalProcessorGUI:
    """
    A class to encapsulate the signal processing and interactive plotting GUI.
    """
    def __init__(self, input_file: str, abs_file: str, output_file: str) -> None:
        """Initializes the processor, loads data, and runs the application."""
        # --- 1. Initialize State and Load Data ---
        self.input_file_name = input_file
        self.abs_time_file_name = abs_file
        self.output_file_name = output_file
        
        self.time: np.ndarray = np.array([])
        self.signal: np.ndarray = np.array([])
        self.merged_indices: np.ndarray = np.array([], dtype=int)
        self.status: List[str] = []
        self.peaks: np.ndarray = np.array([], dtype=int)
        self.minima: np.ndarray = np.array([], dtype=int)

        if not self._load_data():
            return  # Stop initialization if data loading fails

        # --- 2. Signal Processing ---
        self._find_extrema()
        self._identify_breaking_points()

        # --- 3. Create GUI ---
        self._setup_plot()
        self._connect_events()
        self.plot_points()  # Initial plot
        plt.show()

        # --- 4. Save Results on Close ---
        self._save_results()

    def _load_data(self) -> bool:
        """Loads and slices data from input CSV files. Returns False on failure."""
        print("Loading data...")
        try:
            input_df = pd.read_csv(self.input_file_name)
            self.abs_times_df = pd.read_csv(self.abs_time_file_name)
        except FileNotFoundError as e:
            print(f"❌ Error: {e}. Please ensure input files are in the correct directory.")
            return False

        # --- Robustness: Check for required columns ---
        required_input_cols = ['time', 'normalized_APW']
        if not all(col in input_df.columns for col in required_input_cols):
            print(f"❌ Error: Input file '{self.input_file_name}' is missing required columns. Expected: {required_input_cols}")
            return False
        
        if 'Absolute_minima_time' not in self.abs_times_df.columns:
            print(f"❌ Error: Absolute times file '{self.abs_time_file_name}' is missing 'Absolute_minima_time' column.")
            return False

        time_series = input_df['time']
        start_index = time_series.searchsorted(START_TIME_S, side='left')
        end_index = time_series.searchsorted(END_TIME_S, side='right')

        self.time = input_df['time'].iloc[start_index:end_index].to_numpy()
        self.signal = input_df['normalized_APW'].iloc[start_index:end_index].to_numpy()
        print("Data loaded and sliced successfully.")
        return True

    def _filter_by_distance(self, indices: np.ndarray, min_dist: int) -> np.ndarray:
        """Filters indices to ensure a minimum distance between them."""
        if len(indices) == 0:
            return np.array([], dtype=int)
        filtered_indices = [indices[0]]
        for idx in indices[1:]:
            if idx - filtered_indices[-1] >= min_dist:
                filtered_indices.append(idx)
        return np.array(filtered_indices)

    def _find_extrema(self) -> None:
        """Detects peaks and minima in the signal."""
        peaks, _ = find_peaks(self.signal, height=PEAK_HEIGHT_THRESHOLD, prominence=PROMINENCE)
        minima, _ = find_peaks(-self.signal, prominence=PROMINENCE)

        self.peaks = self._filter_by_distance(peaks, MIN_DISTANCE_POINTS)
        self.minima = self._filter_by_distance(minima, MIN_DISTANCE_POINTS)
        self.combined_extrema = np.sort(np.concatenate((self.peaks, self.minima)))

    def _find_closest_candidate(self, t_line: float) -> Optional[int]:
        """For a given time, find the best candidate for a breaking point."""
        if not (self.time.min() <= t_line <= self.time.max()):
            return None

        time_deltas = np.abs(self.time[self.combined_extrema] - t_line)
        candidate_signals = self.signal[self.combined_extrema]

        time_mask = time_deltas < BP_TIME_WINDOW
        strict_signal_mask = (candidate_signals > BP_SIGNAL_RANGE_STRICT[0]) & (candidate_signals < BP_SIGNAL_RANGE_STRICT[1])
        
        valid_indices_strict = self.combined_extrema[time_mask & strict_signal_mask]

        if len(valid_indices_strict) > 0:
            return valid_indices_strict[np.argmin(np.abs(self.time[valid_indices_strict] - t_line))]
        
        valid_indices_relaxed = self.combined_extrema[time_mask]
        if len(valid_indices_relaxed) > 0:
            return valid_indices_relaxed[np.argmin(np.abs(self.time[valid_indices_relaxed] - t_line))]
            
        return None

    def _identify_breaking_points(self) -> None:
        """Identifies breaking points by finding the closest candidates to absolute times."""
        absolute_times = self.abs_times_df["Absolute_minima_time"].values
        
        # --- Structure: Use helper method to find all breaking points ---
        bp_indices = [idx for t_line in absolute_times if (idx := self._find_closest_candidate(t_line)) is not None]
        self.breaking_points = np.unique(bp_indices)
        
        final_extrema = np.setdiff1d(self.combined_extrema, self.breaking_points)
        self.merged_indices = np.sort(np.concatenate((final_extrema, self.breaking_points)))

        # --- Usage of vectorized np.select for faster and cleaner status assignment ---
        conditions = [
            np.isin(self.merged_indices, self.breaking_points),
            np.isin(self.merged_indices, self.peaks)
        ]
        choices = ['BP', 'MAX']
        self.status = np.select(conditions, choices, default='MIN').tolist()

    def _setup_plot(self) -> None:
        """Configures the Matplotlib figure, axes, and slider."""
        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.fig.subplots_adjust(bottom=0.20, left=0.12, right=0.98, top=0.92)
        
        self.ax.plot(self.time, self.signal, label="Signal", color="blue", linewidth=2)

        absolute_times = self.abs_times_df["Absolute_minima_time"].values
        for t_line in absolute_times:
            if self.time.min() <= t_line <= self.time.max():
                self.ax.axvline(x=t_line, color='red', linestyle='--', alpha=0.7)

        self.regular_points_plot, = self.ax.plot([], [], "o", color="green", markersize=7, label="Extrema (MIN/MAX)")
        self.breaking_points_plot, = self.ax.plot([], [], "o", color="red", markersize=7, label="Breaking Point (BP)")
        self.deleted_points_plot, = self.ax.plot([], [], "o", color="gray", alpha=0.4, markersize=7, label="Deleted")
        
        self.ax.set_title("Signal with Detected Peaks and Minima", fontsize=28)
        self.ax.set_xlabel("Time [s]", fontsize=24, labelpad=15)
        self.ax.set_ylabel("Interference Intensity [a.u]", fontsize=24, labelpad=15)
        self.ax.tick_params(axis='both', which='major', labelsize=20)
        self.ax.grid(True)
        self.ax.legend()
        self.ax.set_xlim(self.time.min(), self.time.min() + VIEWING_WINDOW_S)

        ax_slider = self.fig.add_axes([0.2, 0.02, 0.65, 0.03])
        self.time_slider = Slider(
            ax=ax_slider, label='Time',
            valmin=self.time.min(), valmax=self.time.max() - VIEWING_WINDOW_S,
            valinit=self.time.min(), valstep=0.01, color='lightblue'
        )

    def plot_points(self) -> None:
        """Updates the data of the existing scatter plots for performance."""
        status_np = np.array(self.status)
        green_mask = (status_np == 'MAX') | (status_np == 'MIN')
        red_mask = status_np == 'BP'
        gray_mask = status_np == 'DEL'

        self.regular_points_plot.set_data(self.time[self.merged_indices[green_mask]], self.signal[self.merged_indices[green_mask]])
        self.breaking_points_plot.set_data(self.time[self.merged_indices[red_mask]], self.signal[self.merged_indices[red_mask]])
        self.deleted_points_plot.set_data(self.time[self.merged_indices[gray_mask]], self.signal[self.merged_indices[gray_mask]])
        self.fig.canvas.draw_idle()

    def _on_click(self, event) -> None:
        """Handles mouse clicks to re-classify points."""
        # --- Robustness: Handle clicks when no points exist ---
        if event.inaxes != self.ax or len(self.merged_indices) == 0:
            return

        clicked_time = event.xdata
        time_diffs = np.abs(self.time[self.merged_indices] - clicked_time)
        idx_in_status_array = np.argmin(time_diffs)
        
        if time_diffs[idx_in_status_array] > 0.02: return

        current_status = self.status[idx_in_status_array]
        if current_status in ['MAX', 'MIN']:
            self.status[idx_in_status_array] = 'BP'
        elif current_status == 'BP':
            self.status[idx_in_status_array] = 'DEL'
        elif current_status == 'DEL':
            original_idx = self.merged_indices[idx_in_status_array]
            self.status[idx_in_status_array] = 'MAX' if original_idx in self.peaks else 'MIN'
        
        self.plot_points()

    def _update_slider(self, val: float) -> None:
        """Callback for the time slider to update the view."""
        self.ax.set_xlim(val, val + VIEWING_WINDOW_S)
        self.fig.canvas.draw_idle()

    def _connect_events(self) -> None:
        """Connects matplotlib event handlers to their callback methods."""
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.time_slider.on_changed(self._update_slider)

    def _save_results(self) -> None:
        """Saves the final classified points to a CSV file."""
        print(f"\nSaving results to '{self.output_file_name}'...")
        output_df = pd.DataFrame({
            'Time': self.time[self.merged_indices],
            'Signal': self.signal[self.merged_indices],
            'Status': self.status
        })
        output_df = output_df[output_df['Status'] != 'DEL']
        output_df.to_csv(self.output_file_name, index=False)
        print("Done.")

if __name__ == "__main__":
    # --- Flexibility: Use argparse for command-line file management ---
    parser = argparse.ArgumentParser(
        description="Process and classify signal extrema interactively.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input', 
                        default='preprocessed.csv', 
                        help='Input signal data file.')
    parser.add_argument('-a', '--abs', 
                        default='absolute_times_for_BP_calculated_from_envelope_epoch.csv', 
                        help='Absolute times for breaking points.')
    parser.add_argument('-o', '--output', 
                        default='time_signal_status.csv', 
                        help='Output file for classified points.')
    parser.add_argument('--min-distance-points', type=int, default=20, help='Minimum distance in data points between consecutive extrema.')
    parser.add_argument('--peak-height-threshold', type=float, default=1e-8, help='Threshold for peak detection.')
    parser.add_argument('--prominence', type=float, default=0.02511, help='Prominence for peak and minima detection.')
    parser.add_argument('--start-time-s', type=float, default=0.3538, help='Start time in seconds.')
    parser.add_argument('--end-time-s', type=float, default=5.5, help='End time in seconds.')
    parser.add_argument('--viewing-window-s', type=float, default=0.5, help='The length of the viewing window in seconds.')
    parser.add_argument('--bp-time-window', type=float, default=0.05, help='Time window (in seconds) around an absolute time line to look for a BP.')
    parser.add_argument('--bp-signal-range-strict', type=float, nargs=2, default=[0.2, 0.8], metavar=('MIN', 'MAX'), help='Strict signal amplitude range for a point to be a BP (two floats: min max).')
    args = parser.parse_args()

    # Assign command-line arguments to global variables (no global statement needed at module level)
    MIN_DISTANCE_POINTS = args.min_distance_points
    PEAK_HEIGHT_THRESHOLD = args.peak_height_threshold
    PROMINENCE = args.prominence
    START_TIME_S = args.start_time_s
    END_TIME_S = args.end_time_s
    VIEWING_WINDOW_S = args.viewing_window_s
    BP_TIME_WINDOW = args.bp_time_window
    BP_SIGNAL_RANGE_STRICT = tuple(args.bp_signal_range_strict)

    os.system('cls' if os.name == 'nt' else 'clear')
    app = SignalProcessorGUI(input_file=args.input, abs_file=args.abs, output_file=args.output)