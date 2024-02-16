import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss, l1_loss
from TrackSoundDataset import TrackSoundDataset
from cnn import CNNNetwork

# Parameters
TEST_ANNOTATIONS_FILE = "/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/all_spotify_data_output_test.csv"
TEST_AUDIO_DIR = "/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/wav"
BATCH_SIZE = 128
SAMPLE_RATE = 44100
NUM_SAMPLES = 200



def pad_collate(batch):
    """Pads each batch of variable length spectrograms to the longest one in the batch."""
    # Find the maximum length in this batch
    max_length = max([item[0].shape[2] for item in batch])  # item[0] is the spectrogram tensor

    # Pad each spectrogram to the max_length
    padded_batch = []
    labels = []
    for (spectrogram, label) in batch:
        # Amount of padding
        padding_needed = max_length - spectrogram.shape[2]
        # Pad the spectrogram
        padded_spectrogram = torch.nn.functional.pad(spectrogram, (0, padding_needed), "constant", 0)
        padded_batch.append(padded_spectrogram)
        labels.append(label)

    # Stack all padded spectrograms and labels
    padded_spectrograms_stack = torch.stack(padded_batch)
    labels_stack = torch.tensor(labels)

    return padded_spectrograms_stack, labels_stack

# Assuming the same transformation as the training dataset
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=512,
    hop_length=256,
    n_mels=40,
)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Test dataset and dataloader
test_dataset = TrackSoundDataset(
    TEST_ANNOTATIONS_FILE,
    TEST_AUDIO_DIR,
    mel_spectrogram,
    SAMPLE_RATE,
    NUM_SAMPLES,
    device
)
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)

# Model
model = CNNNetwork().to(device)
TRAIN_OUTPUT = "/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/outputs/song.pth"  # Update path as necessary
model.load_state_dict(torch.load(TRAIN_OUTPUT))

# Evaluation function with additional metrics
def evaluate_model(model, test_data_loader, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_samples = 0

    with torch.no_grad():
        for input, target in test_data_loader:
            input, target = input.to(device), target.to(device)
            prediction = model(input).squeeze()
            loss = mse_loss(prediction, target.float())
            mae = l1_loss(prediction, target.float(), reduction='sum')
            total_loss += loss.item() * input.size(0)
            total_mae += mae.item()
            num_samples += input.size(0)

    average_loss = total_loss / num_samples
    average_mae = total_mae / num_samples
    rmse = np.sqrt(average_loss)

    print(f"Test Set Metrics:\nMSE: {average_loss:.4f}\nRMSE: {rmse:.4f}\nMAE: {average_mae:.4f}")



# Evaluate the model
evaluate_model(model, test_data_loader, device)
