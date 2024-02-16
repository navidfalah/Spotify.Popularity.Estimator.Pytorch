import torch
import torchaudio
from cnn import CNNNetwork
from TrackSoundDataset import TrackSoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

def load_model(model_path):
    model = CNNNetwork()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

def prepare_dataset():
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100,  # Make sure this matches your actual audio sample rate
        n_fft=512,
        hop_length=256,
        n_mels=40,
    )
    return TrackSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, "cpu")

def make_inference(model, dataset, index):
    input, target = dataset[index][0], dataset[index][1]
    input.unsqueeze_(0)
    output = model(input)
    predicted_class = torch.argmax(output, dim=1)
    return predicted_class.item(), target

if __name__ == "__main__":
    cnn = load_model("/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/outputs/song.pth")
    print(cnn)
    usd = prepare_dataset()
    print(len(usd))
    predicted, expected = make_inference(cnn, usd, 2)
    print(f"Predicted: '{predicted}', expected: '{expected}'")
