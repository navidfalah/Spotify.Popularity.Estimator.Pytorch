import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from TrackSoundDataset import TrackSoundDataset
from cnn import CNNNetwork
from torch.utils.data import random_split

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
ANNOTATIONS_FILE = "/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/all_spotify_data_output.csv"
AUDIO_DIR = "/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/wav"
TRAIN_OUTPUT = "/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/outputs/song.pth"
ANNOTATIONS_TEST = '/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/all_spotify_data_output_test.csv'
SAMPLE_RATE = 44100
NUM_SAMPLES = 996

def validate_model(model, data_loader, loss_fn, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient computation
        for input_data, target in data_loader:
            input_data, target = input_data.to(device), target.to(device)
            prediction = model(input_data).squeeze()
            loss = loss_fn(prediction, target.float())
            total_loss += loss.item()
            num_batches += 1
    average_loss = total_loss / num_batches
    return average_loss


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()  # Set the model to training mode
    total_loss = 0.0
    num_batches = 0
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        optimiser.zero_grad()
        prediction = model(input).squeeze()  # Squeeze model output to remove singleton dimension
        loss = loss_fn(prediction, target.float())  # Ensure target is float
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        num_batches += 1
        print(f"Predicted: '{prediction}', expected: '{target}'")
    average_loss = total_loss / num_batches
    print(f"Average loss after single epoch: {average_loss:.4f}")  # Print average loss for each epoch


def train(model, data_loader, validation_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        validation_loss = validate_model(model, validation_loader, loss_fn, device)
        print(f"Validation loss after epoch {i+1}: {validation_loss:.4f}")
    print("Finished training")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # Initializing the model
    print("Initializing the model...")
    cnn = CNNNetwork().to(device)
    
    # instantiating our dataset object and create data loader
    print("Loading and preparing the dataset...")
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100,  # Make sure this matches your actual audio sample rate
        n_fft=512,
        hop_length=256,
        n_mels=40,
    )

    usd = TrackSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    
    total_size = len(usd)
    train_size = 796
    test_size = total_size - train_size  # This should be 200
    print(test_size)

    # Randomly split the dataset into training and testing
    train_dataset, test_dataset = random_split(usd, [train_size, test_size])

    # Create DataLoaders for both datasets
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Defining loss function and optimizer
    print("Defining loss function and optimizer...")
    loss_fn = nn.MSELoss()  # Using Mean Squared Error Loss for regression
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    
    # Training the model
    print("Training the model...")
    train(cnn, train_dataloader, test_dataloader, loss_fn, optimiser, device, EPOCHS)
    
    # Saving the trained model
    print("Saving the trained model...")
    torch.save(cnn.state_dict(), TRAIN_OUTPUT)
    print("Trained feed forward net saved at feedforwardnet.pth")
