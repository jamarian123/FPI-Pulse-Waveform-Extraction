# Arterial Pulse Waveform Extraction from Fabry-Perot Interferometric Signals

[cite_start]This repository contains the complete Python-based processing pipeline for extracting Arterial Pulse Waveform (APW) signals from low-finesse Fabry-Perot interferometer (FPI) measurements, as described in the paper: *"[Method for Extracting Arterial Pulse Waveforms from Interferometric Signals](link-to-your-published-paper-here)"*[cite: 1].

[cite_start]A key novelty of this work is the presentation of a complete, accessible Python-based processing pipeline; to the authors' knowledge, no such open-source repository for demodulating these specific interferometric signals to obtain a raw arterial pulse waveform previously existed[cite: 2, 3]. [cite_start]This code is provided to enhance reproducibility and foster further innovation in the field of non-invasive cardiovascular diagnostics[cite: 32].

## Signal Processing Pipeline

The methodology is a multi-step process that takes a raw interferometric signal and reconstructs the final APW, representing the displacement of the sensor's membrane.

1.  **Normalization (`001-base-file-check-with-normalization.py`)**: The raw input signal is validated and preprocessed. [cite_start]This includes outlier removal, applying a Butterworth high-pass filter, and performing min-max normalization to produce a clean, standardized signal[cite: 94, 95, 97, 98].
2.  **Rate of Change Analysis (`002-roc_epoch.py`)**: The script calculates the signal's rate of change and its envelope using the Hilbert transform. [cite_start]Peaks in this envelope are detected, and an averaged epoch is created to identify time points corresponding to low rates of change, which are potential breakpoints[cite: 105, 108, 114, 119].
3.  **Interactive Breakpoint Identification (`extr-breakp-sel.py`)**: This interactive script allows the user to visually inspect the signal and classify extrema. [cite_start]The user can confirm or re-classify points as maxima (MAX), minima (MIN), or breakpoints (BP), which represent phase reversals in the signal corresponding to key cardiac events[cite: 127, 129, 134].
4.  [cite_start]**Displacement Calculation (`plot-delta_z.py`)**: The final script takes the classified points and calculates the cumulative displacement ($\Delta z$) of the FPI membrane, reconstructing the final APW by interpolating the result[cite: 142, 146, 147].

## System Requirements

The project relies on standard Python scientific computing libraries.

### Dependencies
- pandas
- numpy
- scipy
- matplotlib
- [cite_start]neurokit2 [cite: 254]

### Installation
[cite_start]You can install all required dependencies using the provided `requirements.txt` file[cite: 253]. It is highly recommended to use a virtual environment.

```bash
pip install -r requirements.txt
