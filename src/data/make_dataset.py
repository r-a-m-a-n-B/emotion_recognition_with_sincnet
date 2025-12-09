import torch
import librosa
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split



def add_white_noise(signal, noise_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    return signal + noise_factor * noise


class EmotionDataset(Dataset):

    def fix_length(self, signal, train = True):
        signal_length = signal.shape[0]
        if signal_length == self.max_len:
            return signal
        elif signal_length > self.max_len:
            start = np.random.randint(0, signal_length - self.max_len+1)
            signal = signal[start:start + self.max_len]
        else :
            pad_needed = self.max_len - signal_length
            if train:
                left= np.random.randint(0, pad_needed+1)
            else:
                left = pad_needed // 2
            right = pad_needed - left
            signal = np.pad(signal, (left, right), mode="constant", constant_values=0.0)
        return signal

    def __init__(self, csv_path="data/flora_voice_dataset/metadata.csv", data_root="data/flora_voice_dataset/",
                 split="train", sample_rate=16000, duration=3.0,
                 val_ratio=0.1, random_state=42,origin=None):

        

        self.data = pd.read_csv(csv_path)
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * duration)

        self.apply_augment = (split == "train")
        # augmentation hyperparameters
        self.noise_factor = 0.02
        self.pitch_shift_range = (-0.5, 0.5)      
        self.time_stretch_range = (0.9, 1.1 )  

        self.file_index = {}
        for root, _, files in os.walk(self.data_root):
            for f in files:
                self.file_index[f] = os.path.join(root, f)
        
        # --- Split the data ---
        if origin is not None:
            training_data = self.data[self.data["origin"] != origin].reset_index(drop=True)
            testing_data = self.data[self.data["origin"] == origin].reset_index(drop=True)
        else:
            training_data = self.data[self.data["split"] == "train"].reset_index(drop=True)
            testing_data = self.data[self.data["split"] == "test"].reset_index(drop=True)

        # Create validation split 
        if "val" not in self.data["split"].unique():
            print("Creating 10% validation split from training data...")
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

        fname = row["file_name"]
        try:
            path = self.file_index[fname]
        except KeyError:
            raise FileNotFoundError(f"Audio file not found: {fname}")


        # Load audio ---
        waveform, sr = librosa.load(path, sr=None, mono=True)
        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)

        # Applying augmentation
        if self.apply_augment:
            amp = np.random.uniform(0.8, 1.2)
            waveform = waveform * amp

            # Noise addition
            if np.random.rand() < 0.8:
                waveform = add_white_noise(waveform, noise_factor=self.noise_factor)

            # Time stretch
            if np.random.rand() < 0.3:
                rate = np.random.uniform(*self.time_stretch_range)
                try:
                    waveform = librosa.effects.time_stretch(y=waveform, rate=rate)
                except Exception as e:
                    print(f"Time stretch error: {e}")

            # Pitch shift
            if np.random.rand() < 0.3:
                n_steps = np.random.uniform(*self.pitch_shift_range)
                waveform = librosa.effects.pitch_shift(y=waveform, sr=self.sample_rate, n_steps=n_steps)

        waveform = self.fix_length(waveform, train=self.apply_augment)
        # Normalize waveform (important!)
        waveform = waveform - waveform.mean()
        rms = np.sqrt(np.mean(waveform**2))
        if rms > 1e-8:
            waveform = waveform / rms
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        label = self.label_map[emotion]

        return waveform, label