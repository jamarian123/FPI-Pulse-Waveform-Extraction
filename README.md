# Arterial Pulse Waveform Extraction from Fabry-Perot Interferometric Signals

This repository contains the complete Python-based processing pipeline for extracting Arterial Pulse Waveform (APW) signals from low-finesse Fabry-Perot interferometer (FPI) measurements, as described in the paper: "...".

A key novelty of this work is the presentation of a complete, accessible Python-based processing pipeline; to the authors' knowledge, no such open-source repository for demodulating these specific interferometric signals to obtain a raw arterial pulse waveform previously existed. This code is provided to enhance reproducibility and foster further innovation in the field of non-invasive cardiovascular diagnostics.

## Signal Processing Pipeline

The methodology is a multi-step process that takes a raw interferometric signal and reconstructs the final APW, representing the displacement of the sensor's membrane.

1.  **Normalization (`001-base-file-check-with-normalization.py`)**: The raw input signal is validated and preprocessed. This includes outlier removal, applying a Butterworth high-pass filter, and performing min-max normalization to produce a clean, standardized signal.
2.  **Rate of Change Analysis (`002-roc_epoch.py`)**: The script calculates the signal's rate of change and its envelope using the Hilbert transform. Peaks in this envelope are detected, and an averaged epoch is created to identify time points corresponding to low rates of change, which are potential breakpoints.
3.  **Interactive Breakpoint Identification (`extr-breakp-sel.py`)**: This interactive script allows the user to visually inspect the signal and classify extrema. The user can confirm or re-classify points as maxima (MAX), minima (MIN), or breakpoints (BP), which represent phase reversals in the signal corresponding to key cardiac events.
4.  **Displacement Calculation (`plot-delta_z.py`)**: The final script takes the classified points and calculates the cumulative displacement ($\Delta z$) of the FPI membrane, reconstructing the final APW by interpolating the result.

## System Requirements

The project relies on standard Python scientific computing libraries.

### Dependencies
- pandas
- numpy
- scipy
- matplotlib
- neurokit2 

### Installation
You can install all required dependencies using the provided `requirements.txt` file. It is highly recommended to use a virtual environment.

```bash
pip install -r requirements.txt
