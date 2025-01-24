import torch
import numpy as np

def calculate_erd_ers(x: torch.Tensor, fs: int = 128, freq_band=(4, 38)):
    """
    Args:
        x (torch.Tensor): Input EEG signal, shape (B, E, L).
        fs (int): Sampling frequency.
        freq_band (tuple): Frequency band for ERD/ERS computation, default (4, 38) Hz.
    
    Returns:
        erd_ers_features (torch.Tensor): ERD/ERS features, shape (B, E, L).
    """
    # Apply bandpass filtering (using Band-pass filter as an example here)
    from scipy.signal import butter, filtfilt

    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=-1)

    # Get the signal within the frequency range
    filtered_signal = bandpass_filter(x.numpy(), freq_band[0], freq_band[1], fs)

    # ERD/ERS calculation: calculate the power variation at each time point
    erd_ers_features = torch.tensor(filtered_signal)  # The signal should be processed according to the ERD/ERS computation method
    return erd_ers_features

def calculate_erd_ers_loss(erd_ers_real, erd_ers_generated):
    """
    Args:
        erd_ers_real (torch.Tensor): Real ERD/ERS features, shape (B, E, L).
        erd_ers_generated (torch.Tensor): Generated ERD/ERS features, shape (B, E, L).
    
    Returns:
        loss (torch.Tensor): ERD/ERS loss value.
    """
    # Calculate the MSE loss between ERD/ERS values
    return torch.mean((erd_ers_real - erd_ers_generated) ** 2)







