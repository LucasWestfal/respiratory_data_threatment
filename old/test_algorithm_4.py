import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt

def simulate_data(duration=60, sampling_rate=30):
    """
    Simulates a noisy 0.3Hz oscillation signal for a given duration and sampling rate.
    
    Parameters:
    duration (int): Duration of the signal in seconds.
    sampling_rate (int): Sampling rate in Hz.
    
    Returns:
    t (numpy.ndarray): Time vector.
    signal (numpy.ndarray): The simulated signal.
    """
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    frequency = 0.3  # 0.3Hz frequency
    signal = np.sin(2 * np.pi * frequency * t)  # Sine wave at 0.3Hz
    noise = np.random.normal(0, 0.5, signal.shape)  # Gaussian noise
    noisy_signal = signal + noise
    return t, noisy_signal

def filter_frequencies(signal, fs, band_mask, n_most_significant):
    """
    Filter frequencies of a signal based on power spectral density.

    Parameters:
    - signal: 1D array-like
        The input signal.
    - fs: float
        The sampling frequency of the signal.
    - band_mask: tuple
        A tuple representing the frequency band to keep, e.g., (low_cutoff, high_cutoff).
    - n_most_significant: int
        Number of most significant frequencies to keep.

    Returns:
    - filtered_signal: 1D array
        The filtered signal in the specified frequency band.
    - filtered_frequencies: 1D array
        The frequencies corresponding to the n most significant peaks in power spectral density.
    - filtered_psd: 1D array
        The power spectral density values corresponding to the n most significant peaks.
    """

    # Compute the power spectral density using Welch method
    f, psd = welch(signal, fs=fs, nperseg=len(signal))

    # Apply band mask
    mask = (f >= band_mask[0]) & (f <= band_mask[1])
    filtered_frequencies = f[mask]
    filtered_psd = psd[mask]

    # Select the n most significant frequencies
    indices = np.argsort(filtered_psd)[-n_most_significant:]
    filtered_frequencies = filtered_frequencies[indices]
    filtered_psd = filtered_psd[indices]

    # Create a bandpass filter to retain only the frequencies in the specified band
    low_cutoff, high_cutoff = band_mask
    b, a = butter(N=4, Wn=(low_cutoff, high_cutoff), btype='band', fs=fs)
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal, filtered_frequencies, filtered_psd

def plot_results(original_signal, filtered_signal, f, psd, filtered_frequencies, filtered_psd):
    """
    Plot the original and filtered signals, as well as the power spectral density.

    Parameters:
    - original_signal: 1D array-like
        The original input signal.
    - filtered_signal: 1D array-like
        The filtered signal in the specified frequency band.
    - f: 1D array-like
        The frequency values.
    - psd: 1D array-like
        The power spectral density of the original signal.
    - filtered_frequencies: 1D array-like
        The frequencies corresponding to the n most significant peaks in power spectral density.
    - filtered_psd: 1D array-like
        The power spectral density values corresponding to the n most significant peaks.
    """

    # Plot the original and filtered signals
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(original_signal)) / fs, original_signal, label='Original Signal')
    plt.plot(np.arange(len(filtered_signal)) / fs, filtered_signal, label='Filtered Signal')
    plt.title('Original and Filtered Signals')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plot the power spectral density
    plt.subplot(2, 1, 2)
    plt.plot(f, psd, label='Original PSD')
    plt.plot(filtered_frequencies, filtered_psd, 'ro', label='Filtered PSD Peaks')
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Example usage:
# Generate a sample signal (sine wave with noise)
fs = 1000  # Sampling frequency
t = np.arange(0, 5, 1/fs)  # Time vector from 0 to 5 seconds
f1, f2 = 10, 50  # Frequencies of the sine waves
signal = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t)
noise = 0.2 * np.random.randn(len(t))  # Gaussian noise
noisy_signal = signal + noise

# Define the band mask and the number of most significant frequencies to keep
band_mask = (5, 30)  # Keep frequencies between 5 Hz and 30 Hz
n_most_significant = 2

# Process the signal
filtered_signal, filtered_frequencies, filtered_psd = filter_frequencies(noisy_signal, fs, band_mask, n_most_significant)

# Plot the results
f, psd = welch(noisy_signal, fs=fs, nperseg=len(noisy_signal))
plot_results(noisy_signal, filtered_signal, f, psd, filtered_frequencies, filtered_psd)