import numpy as np
import pandas as pd
import os
import librosa
import h5py
import torch
import gzip

def process_mel(mel, target_len, train):
    T = mel.shape[1]
    # Normalize
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    if T < target_len:
        mel = np.pad(mel, ((0, 0), (0, target_len - T)))
    else:
        if train:
            start = np.random.randint(0, T - target_len)
        else:
            start = (T - target_len) // 2
        mel = mel[:, start:start + target_len]

    return mel

# Load in Mel Spectrogram
folder1 = "C:/Users/User/OneDrive/Documents/ResearchProjectHDF5Files/"
save_folder1 = "C:/Users/User/OneDrive/Documents/Processed Mel Spectrograms/"
max_len = 0
len_array = []
for file in os.listdir(folder1):
    temp_hdf5_file = os.path.join(folder1, file)
    with h5py.File(temp_hdf5_file, "r") as f:
        mel_spectrogram1 = f["mel_spectrogram"]
        mel_spectrogram_data = mel_spectrogram1[()]

        # Process Mel Spectrogram
        mel_spectrogram = process_mel(mel_spectrogram_data, target_len=1200, train=False)

        # Convert to Tensor
        mel_tensor = torch.tensor(mel_spectrogram, dtype=torch.float32)

        # Add Channel Dimension
        mel_tensor = mel_tensor.unsqueeze(0)

        # Export Mel Tensor
        video_file = file.split("_")[1]
        video_file = video_file.split(".")[0]
        temp_save_file = "processed_mel_spectrogram_" + video_file + ".pt"
        export_file = os.path.join(save_folder1, temp_save_file)
        torch.save(mel_tensor, export_file)
        print(file)








