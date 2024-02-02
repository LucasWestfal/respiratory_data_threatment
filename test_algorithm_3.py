import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

'''
essa versão faz filtro de banda e depois coleta as frequẽncias dominantes
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

# Let's update the process_signal function to include both a bandpass filter and the selection of the top N frequencies.
def process_signal(signal, sampling_rate, n_freqs=5, lowcut=0.1, highcut=1.0):
    """
    Processes the signal by applying FFT, a bandpass filter to keep frequencies
    within a specified range, and then selects the top N frequencies based on their amplitude.
    
    Parameters:
    signal (numpy.ndarray): The input signal.
    sampling_rate (int): Sampling rate in Hz.
    n_freqs (int): The number of top frequencies to select.
    lowcut (float): Lower frequency bound for the bandpass filter in Hz.
    highcut (float): Upper frequency bound for the bandpass filter in Hz.
    
    Returns:
    filtered_signal (numpy.ndarray): The bandpass filtered signal.
    top_freqs (numpy.ndarray): The selected top N frequencies.
    top_ampls (numpy.ndarray): The amplitudes of the top N frequencies.
    """
    # Subtracting the mean from the signal to remove the zero-frequency component
    signal -= np.mean(signal)
    
    # FFT
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), 1/sampling_rate)
    fft_ampl = np.abs(fft_result)
    
    # Bandpass filter
    # Create a boolean mask for frequencies within the desired range
    band_mask = (fft_freq >= lowcut) & (fft_freq <= highcut)
    # Apply mask to keep only the desired frequencies
    fft_result_filtered = fft_result * band_mask
    
    # Inverse FFT to get the filtered time domain signal
    filtered_signal = np.fft.ifft(fft_result_filtered)
    
    # Select the top N frequencies
    # Find indices of the top N amplitudes within the bandpass range
    top_indices = np.argsort(fft_ampl[band_mask])[-n_freqs:]
    top_freqs = fft_freq[band_mask][top_indices]
    top_ampls = fft_ampl[band_mask][top_indices]

    # Print the top frequencies and their corresponding amplitudes
    print("Top frequencies:")
    for i in range(n_freqs):
        print("{:.2f}Hz: {:.2f}".format(top_freqs[i], top_ampls[i]))
    
    return filtered_signal, top_freqs, top_ampls

# Now let's test the refactored process_signal function
# Simulate data
duration = 60  # seconds
sampling_rate = 30  # Hz
t, noisy_signal = simulate_data(duration, sampling_rate)

# Test the updated process_signal function
filtered_signal, top_freqs, top_ampls = process_signal(noisy_signal, sampling_rate)

# Plot the original, filtered signal and the top frequencies
plt.figure(figsize=(12, 6))

# Original signal
plt.subplot(3, 1, 1)
plt.title("Original Signal")
plt.plot(t, noisy_signal, label='Original')
plt.legend()

# Filtered signal
plt.subplot(3, 1, 2)
plt.title("Filtered Signal")
plt.plot(t, filtered_signal.real, label='Filtered', color='orange')
plt.legend()

# Top frequencies
plt.subplot(3, 1, 3)
plt.title("Top Frequencies")
plt.stem(top_freqs, top_ampls, 'r', markerfmt='r*')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Return the top frequencies and their corresponding amplitudes
top_freqs, top_ampls
