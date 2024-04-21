import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, precision_score, recall_score, f1_score
import torchaudio
import numpy as np
from cnn import CNNNetwork
from TrackSoundDataset import TrackSoundDataset


MODEL_PATH = '/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/outputs/song_march.pth'
ANNOTATIONS_FILE = "/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/spotify_data_updated.csv"
AUDIO_DIR = "/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/wav"
SAMPLE_RATE = 44100
NUM_SAMPLES = 1967
BATCH_SIZE = 32
metadata_dim = 13  # Adjust based on actual metadata used
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the model
model = CNNNetwork(metadata_dim=metadata_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Define transformation and dataset
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=512, hop_length=256, n_mels=40
)
test_dataset = TrackSoundDataset(
    ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device
)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Evaluation function
def evaluate_model(model, data_loader, device):
    all_predictions = []
    all_targets = []
    total_batches = len(data_loader)
    batch_counter = 0

    with torch.no_grad():
        for input_data, metadata, target, _ in data_loader:
            batch_counter += 1
            input_data, metadata, target = input_data.to(device), metadata.to(device), target.to(device)
            predictions = model(input_data, metadata).squeeze()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            if batch_counter % 10 == 0:
                print(f"Processed {batch_counter}/{total_batches} batches.")

    return all_predictions, all_targets

# Evaluate the model
predictions, targets = evaluate_model(model, test_dataloader, device)

# Calculate MAE and R2 Score
mae = mean_absolute_error(targets, predictions)
r2 = r2_score(targets, predictions)
print(f'MAE: {mae:.4f}')
print(f'R2 Score: {r2:.4f}')

# Define a threshold to binarize predictions for classification metrics
threshold = 70

# Binarize predictions and targets based on threshold
predictions_binary = np.array(predictions) >= threshold
targets_binary = np.array(targets) >= threshold

# Calculate classification metrics
precision = precision_score(targets_binary, predictions_binary)
recall = recall_score(targets_binary, predictions_binary)
f1 = f1_score(targets_binary, predictions_binary)

# Calculate accuracy
accuracy = np.mean(predictions_binary == targets_binary)

# Print out all metrics
print(f'Accuracy (with threshold={threshold}): {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
