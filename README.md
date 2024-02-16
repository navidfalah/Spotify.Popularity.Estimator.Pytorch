# Spotify Popularity Estimator Pytorch 🌊🎶🧠

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

The `all_spotify_data_output.csv` file is populated with data obtained from a dedicated project on data capturing, available at [Spotify Popularity Estimator DataCapture](https://github.com/navidfalah/Spotify-Popularity-Estimator-DataCapture). It should contain the following columns:
- `name`: The title of the track.
- `popularity`: The popularity score of the track on Spotify.
- Other audio features like `acousticness`, `danceability`, `energy`, etc.
- `music_file`: The filename of the corresponding audio file in WAV format.

Sample CSV Data:

```
name, popularity, duration_ms, ..., music_file
Higher Power, 73, 211295, ..., 1. Coldplay - Higher Power (128).wav
```

## Installation and Setup 🛠️

1. Clone the repository.
2. Install dependencies from `requirements.txt`.
3. Place your `.wav` files in `data_spotify/wav`.
4. Ensure your `all_spotify_data_output.csv` is in `data_spotify`.

## Usage 🚀

Execute `train.py` to start training the model with your data. Make sure to activate your Python environment!

## Contributing 🤝

Feel free to fork the project, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## Credits & Acknowledgements 👏

- This project was inspired by the deep learning community and music enthusiasts worldwide.
- Special thanks to the developers of PyTorch for their amazing deep learning library.

## License
This project is licensed under the [MIT License](LICENSE.md).
