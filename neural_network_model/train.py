import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from TrackSoundDataset import TrackSoundDataset
from cnn import CNNNetwork
from torch.utils.data import random_split
import syslog
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from constants import (
    ANNOTATIONS_FILE, AUDIO_DIR, TRAIN_OUTPUT,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, SAMPLE_RATE, NUM_SAMPLES
)
s c
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def validate_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for input_data, metadata, target in data_loader:
            input_data, metadata, target = input_data.to(device), metadata.to(device), target.to(device)
            prediction = model(input_data, metadata).squeeze()
            loss = loss_fn(prediction, target.float())
            total_loss += loss.item()
            num_batches += 1
            all_predictions.extend(prediction.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    average_loss = total_loss / num_batches
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    return average_loss, mae, r2

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for input_data, metadata, target in data_loader:
        input_data, metadata, target = input_data.to(device), metadata.to(device), target.to(device)
        optimiser.zero_grad()
        prediction = model(input_data, metadata).squeeze()
        loss = loss_fn(prediction, target.float())
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        num_batches += 1
    average_loss = total_loss / num_batches
    print(f"Training loss: {average_loss:.4f}")

def train(model, data_loader, validation_loader, loss_fn, optimiser, device, epochs, early_stopping_patience=10):
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    
    for epoch in range(epochs):
        syslog.syslog(syslog.LOG_INFO, f"Epoch {epoch+1}/{epochs}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        validation_loss, mae, r2 = validate_model(model, validation_loader, loss_fn, device)
        syslog.syslog(syslog.LOG_INFO, f"Validation Loss: {validation_loss:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        early_stopping(validation_loss)
        if early_stopping.early_stop:
            syslog.syslog(syslog.LOG_INFO, "Early stopping")
            break

    syslog.syslog(syslog.LOG_INFO, "Finished training")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    syslog.syslog(syslog.LOG_INFO, f"Using {device}")
    
    metadata_dim = 15  # Adjust based on actual metadata used

    syslog.syslog(syslog.LOG_INFO, "Initializing the model...")
    cnn = CNNNetwork(metadata_dim=metadata_dim).to(device)
    
    syslog.syslog(syslog.LOG_INFO, "Loading and preparing the dataset...")
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=512, hop_length=256, n_mels=40)

    usd = TrackSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    
    total_size = len(usd)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(usd, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    syslog.syslog(syslog.LOG_INFO, "Defining loss function and optimizer...")
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    
    syslog.syslog(syslog.LOG_INFO, "Training the model...")
    train(cnn, train_dataloader, test_dataloader, loss_fn, optimiser, device, EPOCHS, early_stopping_patience=10)
    
    syslog.syslog(syslog.LOG_INFO, "Saving the trained model...")
    torch.save(cnn.state_dict(), TRAIN_OUTPUT)
    syslog.syslog(syslog.LOG_INFO, "Trained model saved at " + TRAIN_OUTPUT)
