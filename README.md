# SoundWave Neural Network Model 🌊🎶🧠

Welcome to the SoundWave Neural Network Model project! This innovative codebase utilizes the power of Convolutional Neural Networks (CNNs) to analyze and predict the popularity of songs based on various audio features.

## What's Inside? 🧐

- `data_spotify`: Your go-to folder containing all the Spotify song data and WAV files for model training.
- `neural_network_model`: The core of our project with Python files defining our CNN and training procedures.
- `outputs`: Here's where the magic happens - trained models and their outputs will be stored here.
- `utils`: A collection of utility scripts to convert, clean, and manage our data.

## How It Works 🔍

1. **CNN Architecture**: `cnn.py` defines a CNN with multiple convolutional layers, ReLU activations, and pooling to extract features from audio files.
2. **Data Management**: `TrackSoundDataset.py` manages audio data, converting files to the necessary format, and fetching items for processing.
3. **Training**: `train.py` runs the training sessions, validates model performance, and logs the results.

## Preparing Your Data 📊🎵

The `all_spotify_data_output.csv` file should contain the following columns:
- `name`: The title of the track.
- `popularity`: The popularity score of the track on Spotify.
- Other audio features like `acousticness`, `danceability`, `energy`, etc.
- `music_file`: The filename of the corresponding audio file in WAV format.

Sample CSV Data:

```name, popularity, duration_ms, ..., music_file
Higher Power, 73, 211295, ..., 1. Coldplay - Higher Power (128).wav```

