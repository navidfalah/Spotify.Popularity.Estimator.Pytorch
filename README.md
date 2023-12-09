# Audio Neural Processing for Spotify Songs

## Overview
This project is an extension of the Spotify Data Collector, focusing on advanced signal processing of the captured songs using PyTorch and torchaudio. It leverages the data and features obtained from [Spotify Popularity Estimator Data Capture](https://github.com/navidfalah/Spotify-Popularity-Estimator-DataCapture). The primary objective is to preprocess the audio data, apply neural network models for analysis, and predict the efficiency of songs.

## Features
- **Data Cleaning:** Cleanses and aligns data where song names did not match initially.
- **Format Conversion:** Converts audio files from MP3 to WAV for enhanced processing with torchaudio.
- **Neural Network Processing:** Includes a class to fetch data from the database, process the signal through a neural network, and predict song efficiency.
- **Model Training:** The project contains a trained model, with plans for further refinement and development.
- **Future Enhancements:** Plans to increase neural network layers, remove noise, improve song quality, and incorporate additional features from Spotify data for more accurate predictions.
- **Optimized for Linux:** The codebase is tailored and tested to work more efficiently in Linux environments.

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- torchaudio
- A Linux-based operating system for optimal performance
- An environment compatible with CUDA for GPU acceleration (optional)

### Installation and Setup
1. **Clone the repository:**
   ```
   git clone [repository link]
   ```
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
   Dependencies include:
   ```
   [List of Dependencies]
   ```

### Running the Application
To run the application, use the provided scripts. For example, to convert audio files:
```
python cleaner.py
```

## Usage
The project is used to process and analyze audio files. It involves cleaning, converting, and feeding the data through neural networks to evaluate and predict song quality and efficiency.

## Contributing
Contributions for improving and expanding the project are welcome. Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the [MIT License](LICENSE.md).
