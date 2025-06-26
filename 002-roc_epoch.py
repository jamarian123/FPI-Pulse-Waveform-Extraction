import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, find_peaks
import neurokit2 as nk
from matplotlib.backend_bases import MouseButton

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("signal_processing.log"), # Log to a file
        logging.StreamHandler()                      # Also log to the console
    ]
)

# === Helper Functions (Static, do not depend on class state) ===

def butterworth_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a zero-phase Butterworth bandpass filter to the input signal.

    Args:
        data (np.ndarray): The input signal array.
        lowcut (float): The low cutoff frequency in Hz.
        highcut (float): The high cutoff frequency in Hz.
        fs (int): The sampling rate of the signal in Hz.
        order (int, optional): The order of the filter. Defaults to 4.

    Returns:
        np.ndarray: The bandpass-filtered signal.
    """
    nyquist = 0.5 * fs
    normal_lowcut = lowcut / nyquist
    normal_highcut = highcut / nyquist
    b, a = butter(order, [normal_lowcut, normal_highcut], btype='band', analog=False)
    return filtfilt(b, a, data)

def calculate_envelope(signal):
    """
    Calculate the normalized envelope of a signal using the Hilbert transform.

    Args:
        signal (np.ndarray): The input signal.

    Returns:
        np.ndarray: The normalized envelope of the signal.
    """
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    # Avoid division by zero if the signal is flat
    min_val, max_val = np.min(envelope), np.max(envelope)
    if max_val == min_val:
        return envelope - min_val
    return (envelope - min_val) / (max_val - min_val)

# === Core Signal Processing Class ===

class SignalProcessor:
    """
    Encapsulates the entire signal processing pipeline from loading to saving.
    """
    # --- Constants that are less likely to be changed via CLI ---
    FILTER_ORDER = 4
    PADDING_SIZE = 1000
    SKIP_BEGINNING_SAMPLES = 111
    SKIP_END_SAMPLES = 111
    PLOT_FIGURE_SIZE = (12, 7)
    PLOT_LINE_COLOR = "blue"
    PLOT_LINE_WIDTH = 1.5

    def __init__(self, args):
        """
        Initializes the processor with configuration from command-line arguments.
        
        Args:
            args (argparse.Namespace): Parsed arguments from the command line.
        """
        self.args = args
        self.time = None
        self.intensity = None
        self.smoothed_envelope = None
        self.event_peaks = None

    def load_and_prepare_data(self):
        """
        Loads and prepares signal data from the input file specified in args.
        
        Returns:
            bool: True if data was loaded successfully, False otherwise.
        """
        logging.info(f"Attempting to load data from '{self.args.input}'")
        try:
            input_df = pd.read_csv(self.args.input)
            
            if "time" not in input_df.columns or "normalized_APW" not in input_df.columns:
                logging.error("CSV must contain 'time' and 'normalized_APW' columns.")
                return False
                
            self.time = input_df["time"].iloc[self.SKIP_BEGINNING_SAMPLES:-self.SKIP_END_SAMPLES].reset_index(drop=True)
            self.intensity = input_df["normalized_APW"].iloc[self.SKIP_BEGINNING_SAMPLES:-self.SKIP_END_SAMPLES].reset_index(drop=True)
            
            logging.info(f"Successfully loaded {len(self.time)} data points.")
            
            if not self.args.no_plots:
                plt.figure(figsize=self.PLOT_FIGURE_SIZE)
                plt.plot(self.time, self.intensity, color=self.PLOT_LINE_COLOR, linewidth=self.PLOT_LINE_WIDTH, label="Intensity")
                plt.title("Intensity vs Time")
                plt.xlabel("Time (s)")
                plt.ylabel("Intensity")
                plt.grid(True, linestyle="--", linewidth=0.5)
                plt.legend()
                plt.show()
            
            return True
            
        except FileNotFoundError:
            logging.error(f"The file '{self.args.input}' was not found.")
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred during file loading: {e}")
            return False

    def find_events(self):
        """Calculates the smoothed envelope and finds event peaks."""
        logging.info("Calculating rate of change and envelope...")
        rate_of_change = np.gradient(self.intensity, self.time)
        
        padded_roc = np.pad(rate_of_change, self.PADDING_SIZE, mode="reflect")
        padded_envelope = calculate_envelope(padded_roc)
        envelope = padded_envelope[self.PADDING_SIZE:-self.PADDING_SIZE]
        
        logging.info(f"Applying bandpass filter ({self.args.low_cutoff} Hz - {self.args.high_cutoff} Hz)...")
        self.smoothed_envelope = butterworth_bandpass_filter(
            envelope, 
            self.args.low_cutoff, 
            self.args.high_cutoff, 
            self.args.sampling_rate, 
            order=2
        )
        
        logging.info("Detecting peaks in the smoothed envelope...")
        results = nk.signal_findpeaks(self.smoothed_envelope)
        peak_indices = results["Peaks"]

        # Manually filter peaks by height and distance
        filtered_peaks = []
        for idx in peak_indices:
            if self.smoothed_envelope[idx] >= self.args.peak_height:
                if len(filtered_peaks) == 0 or (idx - filtered_peaks[-1] >= self.args.peak_dist):
                    filtered_peaks.append(idx)
        
        self.event_peaks = np.array(filtered_peaks)
        logging.info(f"Found {len(self.event_peaks)} events after filtering.")

        if not self.args.no_plots:
            plt.figure(figsize=(14, 6))
            plt.plot(self.time, self.smoothed_envelope, label="Smoothed Envelope", color=self.PLOT_LINE_COLOR)
            plt.scatter(self.time.iloc[self.event_peaks], self.smoothed_envelope[self.event_peaks], color="red", label="Filtered Peaks")
            plt.title("Smoothed Envelope with Filtered Peaks")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def get_refined_minima(self):
        """Creates averaged epochs and allows interactive refinement of minima."""
        logging.info("Creating and averaging epochs...")
        event_onsets = nk.events_create(event_onsets=self.event_peaks)
        epochs = nk.epochs_create(
            self.smoothed_envelope, event_onsets, sampling_rate=self.args.sampling_rate, 
            epochs_start=self.args.epoch_start, epochs_end=self.args.epoch_end
        )
        average_epoch = nk.epochs_average(epochs)
        
        epoch_time = np.linspace(self.args.epoch_start, self.args.epoch_end, len(average_epoch))

        logging.info(f"Detecting initial {self.args.num_minima} minima from averaged epoch.")
        inverted_signal = -average_epoch["Signal_Mean"]
        minima_indices, _ = find_peaks(inverted_signal)
        minimal_minima = average_epoch["Signal_Mean"].iloc[minima_indices].nsmallest(self.args.num_minima)
        
        minima_times_list = list(epoch_time[minimal_minima.index])
        minima_values_list = list(minimal_minima.values)

        if not self.args.no_plots:
            logging.info("Opening interactive plot to refine minima. Close the plot window to continue.")
            fig, ax = plt.subplots(figsize=self.PLOT_FIGURE_SIZE)

            def redraw():
                ax.clear()
                ax.plot(epoch_time, average_epoch["Signal_Mean"], label="Averaged Signal")
                ax.fill_between(epoch_time, average_epoch["Signal_CI_low"], average_epoch["Signal_CI_high"], 
                                color=self.PLOT_LINE_COLOR, alpha=0.2)
                ax.scatter(minima_times_list, minima_values_list, color="red", zorder=5)
                for t, v in zip(minima_times_list, minima_values_list):
                    ax.annotate(f"{t:.4f}", (t, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
                ax.set_xlabel("Time (s)"), ax.set_ylabel("Signal Amplitude")
                ax.set_title("Averaged Epoch (Click to add/remove, close window to continue)"), ax.legend()
                fig.canvas.draw()

            def onclick(event):
                if event.inaxes != ax: return
                threshold_x = 0.01 * (epoch_time[-1] - epoch_time[0])
                threshold_y = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                
                # Check for removal
                removed = False
                for i, (t, v) in enumerate(zip(minima_times_list, minima_values_list)):
                    if abs(t - event.xdata) < threshold_x and abs(v - event.ydata) < threshold_y:
                        minima_times_list.pop(i), minima_values_list.pop(i)
                        removed = True
                        break
                if not removed: # Add new point
                    idx = (np.abs(epoch_time - event.xdata)).argmin()
                    minima_times_list.append(epoch_time[idx]), minima_values_list.append(average_epoch["Signal_Mean"].iloc[idx])
                redraw()

            redraw()
            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()

        logging.info(f"User finalized {len(minima_times_list)} minima for each event.")
        return np.array(minima_times_list)

    def map_minima_and_save(self, relative_minima_times):
        """Maps relative minima to absolute times and saves them to a CSV file."""
        logging.info("Mapping relative minima to absolute times...")
        all_absolute_minima_times = []
        for onset_sample in self.event_peaks:
            event_time = onset_sample / self.args.sampling_rate
            all_absolute_minima_times.extend(event_time + relative_minima_times)

        df_abs_times = pd.DataFrame({"Absolute_minima_time": all_absolute_minima_times})
        df_abs_times.to_csv(self.args.output, index=False)
        logging.info(f"Final absolute minima times saved to '{self.args.output}'")

    def run(self):
        """Executes the full signal processing pipeline."""
        if not self.load_and_prepare_data():
            return # Exit if data loading fails
        
        self.find_events()
        if self.event_peaks is None or len(self.event_peaks) == 0:
            logging.warning("No events were found. Cannot proceed to epoching. Exiting.")
            return
            
        final_minima_times = self.get_refined_minima()
        self.map_minima_and_save(final_minima_times)
        logging.info("Pipeline finished successfully.")


def main():
    """
    Parses command-line arguments and runs the signal processing pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Process physiological signals to find, epoch, and analyze events.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    # --- Add arguments ---
    parser.add_argument("--input", type=str, default="preprocessed.csv", help="Path to the input CSV data file.")
    parser.add_argument("--output", type=str, default="absolute_times_for_BP_calculated_from_envelope_epoch.csv", help="Path for the output CSV file.")
    parser.add_argument("--sampling-rate", type=int, default=25000, help="Sampling rate of the signal in Hz.")
    parser.add_argument("--low-cutoff", type=float, default=0.5, help="Bandpass filter low cutoff (Hz) for the envelope.")
    parser.add_argument("--high-cutoff", type=float, default=30.0, help="Bandpass filter high cutoff (Hz) for the envelope.")
    parser.add_argument("--peak-height", type=float, default=0.091, help="Minimum height for a peak in the envelope.")
    parser.add_argument("--peak-dist", type=int, default=400, help="Minimum distance (samples) between peaks.")
    parser.add_argument("--epoch-start", type=float, default=-0.2, help="Start time of the epoch window in seconds.")
    parser.add_argument("--epoch-end", type=float, default=0.8, help="End time of the epoch window in seconds.")
    parser.add_argument("--num-minima", type=int, default=8, help="Initial number of minima to detect in the averaged epoch.")
    parser.add_argument("--no-plots", action="store_true", help="Run in non-interactive mode without displaying any plots.")
    
    args = parser.parse_args()
    
    # --- Instantiate and run the processor ---
    processor = SignalProcessor(args)
    processor.run()

if __name__ == "__main__":
    main()
