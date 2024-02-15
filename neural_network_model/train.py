import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from TrackSoundDataset import TrackSoundDataset
from cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
ANNOTATIONS_FILE = "/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/test_data_100/all_spotify_data_refined_100.csv"
AUDIO_DIR = "/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/test_data_100/wav"
TRAIN_OUTPUT = "/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/outputs/song.pth"
SAMPLE_RATE = 105
NUM_SAMPLES = 105


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):

    for input, target in data_loader:
        input, target = input.to(device), target.to(device).float()
        prediction = model(input)
        loss = loss_fn(prediction, target.unsqueeze(1))  # Ensure target is the correct shape
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print(f"loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,  # Increased for better frequency resolution
        hop_length=512,  # Adjust according to the desired time resolution
        n_mels=128,  # Increased for more detailed frequency resolution
        power=2.0,  # The exponent for the magnitude spectrogram
    )

    print(mel_spectrogram)
    
    usd = TrackSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    
    train_dataloader = create_data_loader(usd, BATCH_SIZE)
    cnn = CNNNetwork().to(device)
    loss_fn = nn.MSELoss()  # Using Mean Squared Error Loss for regression
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)
    torch.save(cnn.state_dict(), TRAIN_OUTPUT)
    print("Trained feed forward net saved at feedforwardnet.pth")
