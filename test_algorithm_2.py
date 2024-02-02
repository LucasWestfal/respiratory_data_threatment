import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

'''
essa versão corta frequencias fora do range especificado e volta pro espaço tradicional
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

def process_signal(signal, sampling_rate, lowcut=0.1, highcut=1.0):
    """
    Processes the signal by applying FFT and a bandpass filter to keep frequencies
    within a specified range.
    
    Parameters:
    signal (numpy.ndarray): The input signal.
    sampling_rate (int): Sampling rate in Hz.
    lowcut (float): Lower frequency bound for the bandpass filter in Hz.
    highcut (float): Upper frequency bound for the bandpass filter in Hz.
    
    Returns:
    filtered_signal (numpy.ndarray): The bandpass filtered signal.
    fft_freq (numpy.ndarray): Frequencies vector from FFT.
    fft_ampl (numpy.ndarray): Amplitude spectrum from FFT.
    """
    # Subtracting the mean from the signal to remove the zero-frequency component
    signal = signal - np.mean(signal)
    
    # FFT
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), 1/sampling_rate)
    
    # Bandpass filter
    # Create a boolean mask for frequencies within the desired range
    band_mask = (fft_freq >= lowcut) & (fft_freq <= highcut)
    # Apply mask to the fft result to keep only the desired frequencies
    fft_result_filtered = fft_result * band_mask
    
    # Inverse FFT to get the filtered time domain signal
    filtered_signal = np.fft.ifft(fft_result_filtered)
    
    # Get the amplitude spectrum for the filtered signal
    fft_ampl = np.abs(fft_result_filtered)
    
    return filtered_signal, fft_freq, fft_ampl

# Now let's test the refactored process_signal function
# Simulate data
duration = 60  # seconds
sampling_rate = 30  # Hz
t, noisy_signal = simulate_data(duration, sampling_rate)

# Process signal with bandpass filter
filtered_signal, fft_freq, fft_ampl = process_signal(noisy_signal, sampling_rate)

# Plot the original and filtered signal
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Original Signal")
plt.plot(t, noisy_signal, label='Original')
plt.legend()

plt.subplot(2, 1, 2)
plt.title("Filtered Signal")
plt.plot(t, filtered_signal.real, label='Filtered', color='orange')  # Only plot the real part
plt.legend()
plt.tight_layout()
plt.show()