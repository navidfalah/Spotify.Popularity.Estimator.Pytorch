# Audio Neural Processing for Spotify Songs

## Overview
This project extends the Spotify Data Collector, focusing on advanced signal processing of captured songs using PyTorch and torchaudio. It utilizes data and features from [Spotify Popularity Estimator Data Capture](https://github.com/navidfalah/Spotify-Popularity-Estimator-DataCapture). The goal is to preprocess audio data, apply neural network models for analysis, and predict song efficiency.

## Features
- **Data Cleaning:** Aligns data where song names did not match initially. The `cleaner` script also deletes records from the database if the song is not found in the directory.
- **Format Conversion:** The `convertor` script converts audio files from MP3 to WAV for enhanced processing with torchaudio.
- **Neural Network Processing:** Includes a class to fetch data from the database, process signals through a neural network, and predict song efficiency.
- **Model Training:** Contains a trained model with plans for further refinement and development.
- **Future Enhancements:** Aims to increase neural network layers, remove noise, improve song quality, and add more features from Spotify data for accurate predictions.
- **Optimized for Linux:** Tailored and tested for efficient performance in Linux environments.

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- torchaudio
- Linux-based operating system for optimal performance
- CUDA-compatible environment for GPU acceleration (optional)

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
To run the application, use the provided scripts. For example:
- To clean data and update the database:
  ```
  python cleaner.py
  ```
- To convert audio files from MP3 to WAV:
  ```
  python convertor.py
  ```

## Usage
This project processes and analyzes audio files, involving cleaning, converting, and feeding data through neural networks to evaluate and predict song quality and efficiency.

## Contributing
Contributions to improve and expand the project are welcome. Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the [MIT License](LICENSE.md).
