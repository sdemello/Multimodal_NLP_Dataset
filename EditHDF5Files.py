import numpy as np
import pandas as pd
import h5py
import os
from natsort import natsorted

# Load Embeddings
embeddings_array = np.load("C:/Users/User/Downloads/embeddings.npy")

# Labels File
labels_file = "C:/Users/User/Downloads/ResearchProjectLabels.csv"
labels_pd = pd.read_csv(labels_file)
result_array = labels_pd.iloc[:, 1:].to_numpy()

hdf5_folder = "C:/Users/User/OneDrive/Documents/ResearchProjectHDF5Files/"
mfccs_folder = "C:/Users/User/OneDrive/Documents/Research Project Audio Preprocessing/Mel Frequency Cepstral Coefficients/"
mel_spectrogram_folder = "C:/Users/User/OneDrive/Documents/Research Project Audio Preprocessing/Mel Spectrograms/"
i = 0
for file in natsorted(os.listdir(hdf5_folder)):
    temp_hdf5_file = os.path.join(hdf5_folder, file)
    print(file)

    # MFCCS
    temp_audio_mfccs_array_filename = "mfccs_cleanedVideo" + str(i + 1) + ".npy"
    temp_audio_mfccs_array_file = os.path.join(mfccs_folder, temp_audio_mfccs_array_filename)
    temp_audio_mfccs_array = np.load(temp_audio_mfccs_array_file)

    # Mel Spectrograms
    temp_audio_mel_spectrogram_array_filename = "mel_spectrogram_cleanedVideo" + str(i + 1) + ".npy"
    temp_audio_mel_spectrogram_array_file = os.path.join(mel_spectrogram_folder, temp_audio_mel_spectrogram_array_filename)
    temp_audio_mel_spectrogram_array = np.load(temp_audio_mel_spectrogram_array_file)

    # Embeddings
    temp_embeddings_array = embeddings_array[i]

    # Video Clip Data
    with h5py.File(temp_hdf5_file, 'r') as f:
        data = f['video_data']
        temp_video_data_array = data[()]

    # Label
    temp_label = result_array[i]

    with h5py.File(temp_hdf5_file, 'w') as f:
        f.create_dataset("mfcc", data=temp_audio_mfccs_array, compression="gzip")
        f.create_dataset("mel_spectrogram", data=temp_audio_mel_spectrogram_array, compression="gzip")
        f.create_dataset("text_embedding", data=temp_embeddings_array, compression="gzip")
        f.create_dataset("label", data=temp_label)
        f.create_dataset("video_data", data=temp_video_data_array, compression="gzip")
        f.attrs['fps'] = 2

    i += 1










