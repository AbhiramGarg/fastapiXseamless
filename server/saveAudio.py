import numpy as np
import soundfile as sf
import torch

def save_audio(audio_array, sample_rate, output_filename):

    if isinstance(audio_array, torch.Tensor):
        audio_array = audio_array.cpu().numpy()

    if not isinstance(audio_array, np.ndarray):
        raise TypeError("audio_array must be a NumPy array")

    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)

    if audio_array.ndim > 2:
        raise ValueError(f"Audio data should be 1D or 2D, but got shape {audio_array.shape}")
    elif audio_array.ndim == 2 and audio_array.shape[0] == 1:
        audio_array = audio_array.squeeze(0)  # Remove the extra dimension for mono audio

    sf.write(output_filename, audio_array, sample_rate, format="WAV")