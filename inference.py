import torch
import torchaudio
from cnn import CNNNetwork
from urbansounddataset import UrbanSoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

def load_model(model_path):
    model = CNNNetwork()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

def prepare_dataset():
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    return UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, "cpu")

def make_inference(model, dataset, index):
    input, target = dataset[index][0], dataset[index][1]
    input.unsqueeze_(0)
    output = model(input)
    predicted_class = torch.argmax(output, dim=1)
    return predicted_class.item(), target

if __name__ == "__main__":
    cnn = load_model("songnet.pth")
    print(cnn)
    usd = prepare_dataset()
    print(len(usd))
    predicted, expected = make_inference(cnn, usd, 300)
    print(f"Predicted: '{predicted}', expected: '{expected}'")
