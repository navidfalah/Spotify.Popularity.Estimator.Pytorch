import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from constants import AUDIO_DIR
import syslog


class TrackSoundDataset(Dataset):
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        syslog.syslog(syslog.LOG_INFO, f"TrackSoundDataset initialized with annotations file {annotations_file}")

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        song_title = row['music_file']
        files = os.listdir(AUDIO_DIR)
        audio_file_path = None
        for file in files:
            if song_title in file:
                audio_file_path = os.path.join(AUDIO_DIR, file)
                break
        if not audio_file_path:
            syslog.syslog(syslog.LOG_ERR, f"No audio file found for song title '{song_title}'")
            raise FileNotFoundError(f"No audio file found for song title '{song_title}'")
        signal, sr = torchaudio.load(audio_file_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        label = row['popularity']
        syslog.syslog(syslog.LOG_INFO, f"Processed song title '{song_title}' with popularity label {label}")
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]
