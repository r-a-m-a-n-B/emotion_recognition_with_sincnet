"""
Train an Emotion Recognition model using SincNet as a feature extractor
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.make_data import EmotionDataset
from src.models.sincnet.model import SincNet


class SincNetEmotion(nn.Module):
    """
    Full model: SincNet feature extractor + classifier head
    """
    def __init__(self, num_classes: int, sample_rate: int = 16000):
        super().__init__()
        self.sincnet = SincNet(sample_rate=sample_rate)

        # Infer output feature dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, sample_rate)
            feat = self.sincnet(dummy)
            feat_dim = feat.shape[1] * feat.shape[2]

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.sincnet(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.classifier(x)


def train_emotion_model(csv_path="data/metadata.csv",
                        data_root="data/raw",
                        epochs=25,
                        batch_size=32,
                        lr=1e-3):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # Load datasets
    train_ds = EmotionDataset(csv_path, data_root, split="train")
    val_ds = EmotionDataset(csv_path, data_root, split="val")

    num_classes = len(train_ds.emotions)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = SincNetEmotion(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for wave, label in train_dl:
            wave, label = wave.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(wave)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch}/{epochs} | Loss: {total_loss/len(train_dl):.4f} | Train Acc: {acc:.2f}%")

        # --- Validation ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for wave, label in val_dl:
                wave, label = wave.to(device), label.to(device)
                preds = torch.argmax(model(wave), dim=1)
                val_correct += (preds == label).sum().item()
                val_total += label.size(0)
        val_acc = 100 * val_correct / val_total
        print(f"→ Validation Accuracy: {val_acc:.2f}%")

    torch.save(model.state_dict(), "models/sincnet_emotion_model.pth")
    print("✅ Model saved at models/sincnet_emotion_model.pth")


if __name__ == "__main__":
    train_emotion_model()
