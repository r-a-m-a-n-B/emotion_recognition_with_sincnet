# import torch

# from src.models.sincnet import SincNet

# if __name__ == "__main__":
#     # Test SincNet
#     model = SincNet()
#     waveforms = torch.randn(1, 1, 16000)
#     outputs = model(waveforms)
#     print(outputs.shape)


import torch
import librosa
import numpy as np
from src.models.train_model import SincNetEmotion
from src.data.make_dataset import EmotionDataset


def predict_emotion(audio_path,
                    checkpoint="models/sincnet_emotion_model.pth",
                    csv_path="data/metadata.csv",
                    sample_rate=16000,
                    duration=3.0):
    """
    Predict the emotion of a given audio file using the trained SincNetEmotion model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset to get label map
    dataset = EmotionDataset(csv_path=csv_path, split="train")
    label_map = dataset.label_map
    inv_label_map = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    # Load trained model
    model = SincNetEmotion(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # Load and preprocess audio
    waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    max_len = int(sample_rate * duration)
    if len(waveform) < max_len:
        waveform = np.pad(waveform, (0, max_len - len(waveform)), mode="constant")
    else:
        waveform = waveform[:max_len]

    waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(waveform)
        pred = torch.argmax(logits, dim=1).item()

    emotion = inv_label_map[pred]
    print(f"Predicted Emotion: {emotion}")
    return emotion


if __name__ == "__main__":
    # Example test: replace with your real audio path
    predict_emotion("data/raw/happy_01.wav")
