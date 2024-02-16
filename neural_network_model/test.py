import torch
import torchaudio
from torch.utils.data import DataLoader
from TrackSoundDataset import TrackSoundDataset
from cnn import CNNNetwork

# Assuming the paths for evaluation data are defined or add them here
EVAL_ANNOTATIONS_FILE = "/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/data.csv"
EVAL_AUDIO_DIR = "/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/wav"
TRAIN_OUTPUT = "/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/outputs/song.pth"
EVAL_NUM_SAMPLES = 5  # Number of samples in your evaluation dataset
SAMPLE_RATE = 44100
BATCH_SIZE = 64


def pad_collate(batch):
    """Pads each batch of variable length spectrograms to the longest one in the batch."""
    print("Padding batch...")  # Logging
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
    print("Batch padded.")  # Logging

    return padded_spectrograms_stack, labels_stack

def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()  # Set the model to evaluation mode
    print("Model set to evaluation mode.")  # Logging
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():  # No need to track gradients for evaluation
        for input_data, target in data_loader:
            input_data, target = input_data.to(device), target.to(device)
           
            prediction = model(input_data).squeeze()
            loss = loss_fn(prediction, target.float())
            total_loss += loss.item()
            num_batches += 1
            print(f"target {target.float()}")
            print(f"prediction {prediction}")
            print(f"Batch {num_batches}: Loss = {loss.item():.4f}")  # Logging per batch
    
    average_loss = total_loss / num_batches
    print(f"Average loss on evaluation data: {average_loss:.4f}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for evaluation")

    # Load the trained model
    cnn = CNNNetwork().to(device)
    cnn.load_state_dict(torch.load(TRAIN_OUTPUT))
    print("Model loaded.")

    # Prepare the evaluation dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100,
        n_fft=512,
        hop_length=256,
        n_mels=40,
    )

    eval_dataset = TrackSoundDataset(
        EVAL_ANNOTATIONS_FILE,
        EVAL_AUDIO_DIR,
        mel_spectrogram,
        SAMPLE_RATE,
        EVAL_NUM_SAMPLES,
        device
    )
    print("Evaluation dataset prepared.")

    eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=True, collate_fn=pad_collate)
    print("Dataloader prepared.")

    # Evaluate the model
    loss_fn = torch.nn.MSELoss()
    print("Starting model evaluation...")
    evaluate_model(cnn, eval_dataloader, loss_fn, device)
