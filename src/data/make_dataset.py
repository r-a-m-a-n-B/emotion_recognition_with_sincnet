import torch
import librosa
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class EmotionDataset(Dataset):

    def __init__(self, csv_path="data/flora_voice_dataset/metadata.csv", data_root="data/flora_voice_dataset/",
                 split="train", sample_rate=16000, duration=3.0,
                 val_ratio=0.1, random_state=42):

        self.data = pd.read_csv(csv_path)
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * duration)

        # --- Split the data ---
        training_data = self.data[self.data["split"] == "train"].reset_index(drop=True)
        testing_data = self.data[self.data["split"] == "test"].reset_index(drop=True)

        # Create validation split 
        if "val" not in self.data["split"].unique():
            print("🟡 Creating 10% validation split from training data...")
            training_data, validation_data = train_test_split(
                training_data,
                test_size=val_ratio,
                stratify=training_data["emotion"],
                random_state=random_state
            )
            training_data["split"] = "train"
            validation_data["split"] = "val"

            # Merge new splits back into full dataset (for internal tracking)
            self.data = pd.concat([training_data, validation_data, testing_data], ignore_index=True)
        else:
            validation_data = self.data[self.data["split"] == "val"].reset_index(drop=True)

        # --- Assign the correct split to this dataset instance ---
        if split == "train":
            self.data = training_data
        elif split == "val":
            self.data = validation_data
        elif split == "test":
            self.data = testing_data
        else:
            raise ValueError("split must be one of ['train', 'val', 'test'].")

        # --- Create emotion label mapping ---
        self.emotions = sorted(self.data["emotion"].unique())
        self.label_map = {emo: idx for idx, emo in enumerate(self.emotions)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = f"{self.data_root}/{row['file_name']}"
        emotion = row["emotion"]

        import glob, os
        if not os.path.exists(path):
            search_pattern = os.path.join(self.data_root, "**", os.path.basename(path))
            matches = glob.glob(search_pattern, recursive=True)
        if len(matches) > 0:
            path = matches[0]
            # print(f"[INFO] Found '{os.path.basename(path)}' in {os.path.dirname(path)}")
        else:
            raise FileNotFoundError(f"Audio file not found: {path}")

        # Load audio ---
        waveform, sr = librosa.load(path, sr=self.sample_rate, mono=True)

        # process audio to fixed length
        if len(waveform) < self.max_len:
            waveform = np.pad(waveform, (0, self.max_len - len(waveform)), mode="constant")
        else:
            waveform = waveform[:self.max_len]

        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        label = self.label_map[emotion]

        return waveform, label










