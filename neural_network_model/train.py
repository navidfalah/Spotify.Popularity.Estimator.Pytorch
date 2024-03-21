import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, random_split
import syslog
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from constants import (
    ANNOTATIONS_FILE, AUDIO_DIR, TRAIN_OUTPUT,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, SAMPLE_RATE, NUM_SAMPLES
)
from TrackSoundDataset import TrackSoundDataset  # Adjusted to return song titles
from cnn import CNNNetwork

class EarlyStopping:
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
                syslog.syslog(syslog.LOG_WARNING, f'EarlyStopping counter: {self.counter} out of {self.patience}')
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
        for input_data, metadata, target, song_titles in data_loader:
            input_data, metadata, target = input_data.to(device), metadata.to(device), target.to(device)
            prediction = model(input_data, metadata).squeeze()
            loss = loss_fn(prediction, target.float())
            total_loss += loss.item()
            num_batches += 1
            all_predictions.extend(prediction.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            for title, pred, actual in zip(song_titles, prediction.cpu().numpy(), target.cpu().numpy()):
                syslog.syslog(syslog.LOG_INFO, f"Validation - Song: {title}, Predicted Popularity: {pred:.2f}, Actual Popularity: {actual}")

    average_loss = total_loss / num_batches
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    syslog.syslog(syslog.LOG_INFO, f'Validation - Avg Loss: {average_loss:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')
    return average_loss, mae, r2

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for input_data, metadata, target, song_titles in data_loader:
        input_data, metadata, target = input_data.to(device), metadata.to(device), target.to(device)
        optimiser.zero_grad()
        prediction = model(input_data, metadata).squeeze()
        loss = loss_fn(prediction, target.float())
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        num_batches += 1
    average_loss = total_loss / num_batches
    syslog.syslog(syslog.LOG_INFO, f"Training loss: {average_loss:.4f}")

def train(model, train_dataloader, test_dataloader, loss_fn, optimiser, device, epochs, early_stopping_patience=10):
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    
    for epoch in range(epochs):
        syslog.syslog(syslog.LOG_INFO, f"Epoch {epoch+1}/{epochs}")
        train_single_epoch(model, train_dataloader, loss_fn, optimiser, device)
        validation_loss, mae, r2 = validate_model(model, test_dataloader, loss_fn, device)
        syslog.syslog(syslog.LOG_INFO, f"Validation Loss: {validation_loss:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        early_stopping(validation_loss)
        if early_stopping.early_stop:
            syslog.syslog(syslog.LOG_INFO, "Early stopping")
            break

    syslog.syslog(syslog.LOG_INFO, "Finished training")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    syslog.syslog(syslog.LOG_INFO, f"Using {device}")

    metadata_dim = 13  # Adjust based on actual metadata used
    cnn = CNNNetwork(metadata_dim=metadata_dim).to(device)
    
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=512, hop_length=256, n_mels=40)
    usd = TrackSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    
    total_size = len(usd)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(usd, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    
    train(cnn, train_dataloader, test_dataloader, loss_fn, optimiser, device, EPOCHS)
    
    torch.save(cnn.state_dict(), TRAIN_OUTPUT)
    syslog.syslog(syslog.LOG_INFO, "Trained model saved at " + TRAIN_OUTPUT)
