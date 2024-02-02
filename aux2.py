import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, periodogram

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
    

    # Select the top N frequencies
    # Find indices of the top N amplitudes within the bandpass range
    duration = signal[-1] - signal[0]
    sampling_freq = 1/sampling_rate
    (top_freqs, Pxx_den) = periodogram(signal, sampling_freq, scaling='density', nfft = 3600)
    new_top_frecs = top_freqs[:-1].copy()# gambiarra pra acertar o tamanho
    new_Pxx_den = Pxx_den[:-1].copy()# gambiarra pra acertar o tamanho
    # Bandpass filter
    # Create a boolean mask for frequencies within the desired range
    band_mask = (fft_freq >= lowcut) & (fft_freq <= highcut)
    
    # Apply mask to keep only the desired frequencies
    fft_result_filtered = fft_result * band_mask

    # top frequencies filter
    top_indices = np.argsort(new_top_frecs[band_mask])[-n_freqs:]
    top_densities = new_Pxx_den[band_mask][top_indices]
    
    mask_top_freqs = []
    for i in range(len(fft_result_filtered)):
        if i in top_indices:
            mask_top_freqs.append(1)
        else:
            mask_top_freqs.append(0)
    
    
    # Print the top frequencies and their corresponding amplitudes
    print("Top frequencies: TA ERRADO AINDA")
    for i in range(n_freqs):
        print("{:.2f}Hz: {:.2f}".format(top_freqs[i], Pxx_den[i]))
    

    # Inverse FFT to get the filtered time domain signal and Apply mask to keep only the desired frequencies

    filtered_signal = np.fft.ifft(fft_result_filtered * mask_top_freqs)
    
    return filtered_signal, top_freqs, top_densities

## TA TUDO ERRADOOOOOOOOOOOOOOOOOO
# em especial, os filtros tao um lixo
# tenho que conseguir selecionar só as top frequencias pra destransformar