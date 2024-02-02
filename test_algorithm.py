import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

'''
 essa versão só pega as frequencias dominantes
'''

# Function to simulate a noisy 0.3 Hz oscillation signal
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

# Function to apply FFT and filter the signal
def process_signal(signal, sampling_rate, top_freq=5):
    """
    Processes the signal by applying FFT and filtering out all but the top 5 frequencies.
    
    Parameters:
    signal (numpy.ndarray): The input signal.
    sampling_rate (int): Sampling rate in Hz.
    top_freq (int): The number of top frequencies to retain.
    
    Returns:
    freqs (numpy.ndarray): Frequencies vector.
    psd (numpy.ndarray): Power Spectral Density of the signal.
    filtered_signal (numpy.ndarray): The filtered signal in time domain.
    """
    # FFT of the signal
    fft_result = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1/sampling_rate)
    psd = np.abs(fft_result) ** 2
    
    # Identify the top frequencies
    top_indices = np.argsort(psd)[-top_freq:]
    top_frequencies = freqs[top_indices]
    
    # Filter design to keep only the top frequencies
    nyquist_rate = sampling_rate / 2.0
    filter_band = [(f - 0.05) / nyquist_rate for f in top_frequencies] + \
                  [(f + 0.05) / nyquist_rate for f in top_frequencies]
    filter_band = np.clip(filter_band, 0, 1)  # Ensure within [0, 1] range
    filter_coefs = np.zeros(len(freqs))
    for i in range(top_freq):
        filter_coefs[(freqs >= filter_band[2*i]) & (freqs <= filter_band[2*i+1])] = 1
    
    # Apply the filter in the frequency domain
    filtered_fft = fft_result * filter_coefs
    filtered_signal = np.fft.irfft(filtered_fft)

    # Print the top frequencies and their corresponding amplitudes
    print("Top frequencies:")
    for i in range(top_freq):
        print("{:.2f}Hz: {:.2f}".format(top_frequencies[i], psd[top_indices[i]]))
    
    return freqs, psd, filtered_signal, top_frequencies

# Function to plot original and filtered signals
def plot_signals(t, original_signal, filtered_signal, title="Signal Filtering"):
    """
    Plots the original and filtered signals on the same graph.
    
    Parameters:
    t (numpy.ndarray): Time vector.
    original_signal (numpy.ndarray): The original, noisy signal.
    filtered_signal (numpy.ndarray): The filtered signal.
    title (str): The title of the plot.
    """
    plt.figure(figsize=(15, 5))
    plt.plot(t, original_signal, label='Original Signal')
    plt.plot(t, filtered_signal, label='Filtered Signal')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Amplitude')
    plt.legend()
    plt.show()

# Simulate data
t, noisy_signal = simulate_data()

# Process signal
sampling_rate = 30  # Sampling rate in Hz
freqs, psd, filtered_signal, top_frequencies = process_signal(noisy_signal, sampling_rate)

# Plot original and filtered signals
plot_signals(t, noisy_signal, filtered_signal)
