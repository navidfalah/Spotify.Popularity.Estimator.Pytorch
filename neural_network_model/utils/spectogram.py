import torchaudio
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use('TkAgg')
# Load your audio file
audio_path = '/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/test_data_100/wav/Alcohol - Marshmello Anuel AA (128).wav'

waveform, sample_rate = torchaudio.load(audio_path)

# Normalize waveform to a fixed scale for consistent volume levels
waveform = waveform / torch.max(torch.abs(waveform))

# If your neural network expects a fixed-size input, ensure the waveform is the correct length
# Trim or pad the waveform to a fixed length if necessary
fixed_length = sample_rate * 5  # for a fixed-length of 5 seconds
if waveform.size(1) < fixed_length:
    # Pad the waveform with zeros at the end
    waveform = torch.nn.functional.pad(waveform, (0, fixed_length - waveform.size(1)))
else:
    # Trim the waveform to the fixed length
    waveform = waveform[:, :fixed_length]

# Define the MelSpectrogram transformation with improved parameters
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=2048,  # Increase if more frequency resolution is needed
    hop_length=512,  # Decrease if more time resolution is needed
    n_mels=128,  # Increase for more detailed frequency resolution
    power=2.0,  # Usually 2.0, the exponent for the magnitude spectrogram
)

# Apply the transformation
mel_spec = mel_spectrogram(waveform)

# Convert to decibels
mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

# Visualizing the Mel Spectrogram with a consistent color scale
min_db = mel_spec_db.min()
max_db = mel_spec_db.max()

plt.figure(figsize=(10, 4))
plt.imshow(mel_spec_db[0].detach().numpy(), cmap='viridis', origin='lower', aspect='auto',
           vmin=min_db, vmax=max_db)  # Set consistent color scale
plt.xlabel('Time (s)')
plt.ylabel('Mel Frequency')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.savefig('your_figure.png')
plt.show()