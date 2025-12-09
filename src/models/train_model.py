"""
Train an Emotion Recognition model using SincNet as a feature extractor
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.make_dataset import EmotionDataset
from src.models.sincnet.model import SincNet
from datetime import datetime
import os

class SincNetEmotion(nn.Module):
    """
    Full model: SincNet feature extractor + classifier head
    """
    def __init__(self, num_classes: int, sample_rate: int = 16000):
        super().__init__()
        self.sincnet = SincNet(sample_rate=sample_rate)

        # Dynamically calculate feature size
        with torch.no_grad():
            dummy = torch.randn(1, 1, 16000*3)  # 3 sec sample
            feat = self.sincnet(dummy)
            flattened_dim = feat.shape[1]
            print(f"[INFO] SincNet output features per sample: {flattened_dim}")

        # --- DNN classifier (as per Ravanelli)
        self.dnn = nn.Sequential(
            nn.Linear(flattened_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.sincnet(x)
        x = torch.mean(x, dim=2)
        #x = x.view(x.size(0), -1)  # flatten
        return self.dnn(x)


def train_emotion_model(csv_path="/content/emotion_recognition_with_sincnet/src/data/flora_voice_dataset/metadata.csv",
                        data_root="/content/emotion_recognition_with_sincnet/src/data/flora_voice_dataset",
                        epochs=50,
                        batch_size=64,
                        lr=2e-3,
                        N_eval_epoch=5,origin=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # Load datasets
    train_ds = EmotionDataset(csv_path, data_root, split="train",origin=origin)
    val_ds = EmotionDataset(csv_path, data_root, split="val",origin=origin)
    test_ds = EmotionDataset(csv_path, data_root, split="test",origin=origin)

    print(f"Number of training samples:   {len(train_ds)}")
    print(f"Number of validation samples: {len(val_ds)}")
    print(f"Number of test samples:       {len(test_ds)}\n")

    num_classes = len(train_ds.emotions)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,num_workers=2, pin_memory=True,persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size,num_workers=2, pin_memory=True, persistent_workers=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size,num_workers=2, pin_memory=True, persistent_workers=True)

    model = SincNetEmotion(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5
    )

    # resuming from checkpoint
    best_val_acc = 0.0
    start_epoch = 1

    if os.path.exists("checkpoint.pth"):
        print("Resuming from checkpoint...")
        checkpoint = torch.load("checkpoint.pth", map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        # Restore best validation accuracy if saved
        best_val_acc = checkpoint.get("best_val_acc", 0.0)


    # Results log file
    os.makedirs("models", exist_ok=True)
    res_path = "models/training_results.res"
    with open(res_path, "w") as f:
        f.write(f"Training Log - {datetime.now()}\n\n")

    for epoch in range(start_epoch, epochs + 1):
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
        scheduler.step()
        train_acc = 100 * correct / total
        train_loss = total_loss / len(train_dl)

        # --- Validation ---
        model.eval()
        val_correct, val_total , val_loss_sum = 0, 0, 0
        with torch.no_grad():
            for wave, label in val_dl:
                wave, label = wave.to(device), label.to(device)
                logits = model(wave)
                val_loss_sum += criterion(logits, label).item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == label).sum().item()
                val_total += label.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss_sum / len(val_dl)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/best_sincnet.pth")
            print("Best model saved!")

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        with open(res_path, "a") as f:
                f.write(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n")
        
        # Full test every few epochs (like mravanelli/SincNet)
        if epoch % N_eval_epoch == 0:
            test_correct, test_total, test_loss_sum = 0, 0, 0
            with torch.no_grad():
                for wave, label in test_dl:
                    wave, label = wave.to(device), label.to(device)
                    logits = model(wave)
                    test_loss_sum += criterion(logits, label).item()
                    preds = torch.argmax(logits, dim=1)
                    test_correct += (preds == label).sum().item()
                    test_total += label.size(0)

            test_acc = 100 * test_correct / test_total
            test_loss = test_loss_sum / len(test_dl)

            print(f"-- [Test @ Epoch {epoch}] Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")

            # Save model and results
            torch.save(model.state_dict(), f"models/sincnet_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, "checkpoint.pth")
            with open(res_path, "a") as f:
                f.write(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%\n")

    print(f"\n Training complete! Logs saved to: {res_path}")


if __name__ == "__main__":
    train_emotion_model()

  

