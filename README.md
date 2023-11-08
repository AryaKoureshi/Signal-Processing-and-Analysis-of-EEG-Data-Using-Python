# Signal Processing and Analysis of EEG Data Using Python

This project demonstrates various signal processing techniques, such as signal generation, window functions, filtering, downsampling, zero-padding, and the application of time-frequency analysis using the Short-Time Fourier Transform (STFT)¹[1]²[2]. The project uses Python and its libraries, such as NumPy, SciPy, and Matplotlib, to implement and visualize the methods. The project also analyzes an EEG signal sampled at a rate of 256 Hz and explores its time-domain, frequency-domain, and time-frequency characteristics³[3].

## Project Structure

The project consists of two main parts:

- **Question 1**: This part covers the basics of signal processing, such as generating a chirp signal, applying different window functions, and performing time-frequency analysis using the STFT¹[1]. It also investigates how different parameters, such as window length, overlapping points, and number of DFT points, affect the time and frequency resolution of the spectrogram⁴[4].
- **Question 2**: This part focuses on the analysis of an EEG signal, which is a type of biomedical signal that measures the electrical activity of the brain. It applies low-pass filtering, downsampling, zero-padding, and the DFT to the EEG signal and examines how these techniques influence the signal's representation and frequency content⁵[5]. It also uses the STFT to provide a comprehensive time-frequency analysis of the EEG signal.

## Project Requirements

The project requires the following Python libraries:

- NumPy: A library for scientific computing and working with arrays.
- SciPy: A library for scientific and technical computing, such as signal processing, linear algebra, optimization, and statistics.
- Matplotlib: A library for creating plots and visualizations.
- Scipy.io: A module for reading and writing MATLAB files.

## Project Files

The project files include:

- **NewEEGSignal.mat**: A MATLAB file that contains the EEG signal data.
- **Analysis_of_EEG_Data.ipynb**: A Jupyter notebook that contains the Python code and the results for both questions.
