import numpy as np
import matplotlib.pyplot as plt

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

# Function definitions
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

    # Compute the FFT of the signal
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1/fs)
    fft_values = np.fft.fft(signal)

    # Apply band mask
    mask = (freqs >= band_mask[0]) & (freqs <= band_mask[1])
    filtered_frequencies = freqs[mask]
    filtered_fft = fft_values[mask]
    print(filtered_frequencies)
    # Compute power spectral density
    psd = np.abs(filtered_fft) ** 2

    # Select the n most significant frequencies
    indices = np.argsort(psd)[-n_most_significant:]
    filtered_frequencies = filtered_frequencies[indices]
    filtered_psd = psd[indices]
    
    # Create a filter in the frequency domain
    low_cutoff, high_cutoff = band_mask
    '''
    fft_values[(freqs < low_cutoff) or (freqs > high_cutoff) or (freqs not in filtered_frequencies)] = 0
    '''
    new_values = []
    for i in range(len(fft_values)):
        if (freqs[i] < low_cutoff) or (freqs[i] > high_cutoff) or (i in indices):
            new_values.append(0)
        else:
            new_values.append(fft_values[i])
    
    filtered_signal = np.fft.ifft(new_values)

    return filtered_signal, filtered_frequencies, filtered_psd

def plot_results(original_signal, filtered_signal, freqs, psd, filtered_frequencies, filtered_psd, fs):
    """
    Plot the original and filtered signals, as well as the power spectral density.

    Parameters:
    - original_signal: 1D array-like
        The original input signal.
    - filtered_signal: 1D array-like
        The filtered signal in the specified frequency band.
    - freqs: 1D array-like
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
    plt.plot(freqs, psd, label='Original PSD')
    plt.plot(filtered_frequencies, filtered_psd, 'ro', label='Filtered PSD Peaks')
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()