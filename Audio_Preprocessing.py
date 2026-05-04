import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import os
import sys
import soundfile as sf
import noisereduce as nr
from df.enhance import enhance, init_df, load_audio, save_audio
from pydub import AudioSegment
import io
from mutagen.mp3 import MP3
from Find_Timestamps_Scene_Changes import *

# Audio Cleaning
def clean_audio_deepfilternet(input_mp3, temp_wav, output_mp3, trim, start_time, end_time):
    # Convert and resample the audio to 48 kHz WAV file (DeepFilterNet's required sample rate)
    audio = AudioSegment.from_mp3(input_mp3)
    if trim is True:
        start_time_ms = start_time*1000
        end_time_ms = end_time*1000
        audio = audio[start_time_ms:end_time_ms]
    audio = audio.set_frame_rate(48000).set_channels(1)
    audio.export(temp_wav, format="wav")

    # Use DeepFilterNet for enhancement
    model, df_state, _ = init_df()
    audio_data, _ = load_audio(temp_wav, sr=df_state.sr())
    enhanced_audio = enhance(model, df_state, audio_data)
    save_audio("enhanced.wav", enhanced_audio, df_state.sr())

    # Convert back to MP3
    AudioSegment.from_wav("enhanced.wav").export(output_mp3, format="mp3")

    # Clean up the temporary file
    os.remove(temp_wav)
    os.remove("enhanced.wav")
    print(f"Enhanced audio saved to {output_mp3}")

# Get Mel Spectrogram and Mel-Frequency Cepstral Coefficients (MFCCs)
def get_mel_spectrogram_and_mfccs(filename, filename_mel_spectrogram_np, filename_mfccs_np):
    # Load audio file
    y, sr = librosa.load(filename)

    # Compute STFT (Short-Time Fourier Transform)
    D = librosa.stft(y)
    # Convert amplitude to dB
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Mel Spectrograms
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Extract MFCCs (default 20 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    """
    # Plotting Mel-Spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, x_axis="time", y_axis="linear", sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-Frequency Power Spectrogram')
    plt.tight_layout()
    plt.show()

    # Plotting Mel-Frequency Cepstral Coefficients
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()
    """
    np.save(filename_mel_spectrogram_np, S_db)
    np.save(filename_mfccs_np, mfccs)

    return S_db, mfccs


folder1 = "C:/Users/User/OneDrive/Documents/Research Project Audio Files/"
video_folder1 = "C:/Users/User/OneDrive/Documents/Research Project Data - Copy/"
trim = False
if trim is True:
    temp_trimmed_wav_file = "C:/Users/User/Downloads/temp_trimmed_file.wav"
    output_trimmed_folder1 = "C:/Users/User/OneDrive/Documents/Research Project Cleaned Trimmed Audio Files/"
    mel_spectrogram_folder = "C:/Users/User/OneDrive/Documents/Research Project Trimmed Audio Preprocessing/Mel Spectrograms/"
    mfccs_folder = "C:/Users/User/OneDrive/Documents/Research Project Trimmed Audio Preprocessing/Mel Frequency Cepstral Coefficients/"
    i = 0
    for file in os.listdir(folder1):
        print(f"{i+1}) {file}")
        temp_audio_file = os.path.join(folder1, file)

        output_filename = "cleanedTrimmed" + str(file.split(".")[0]) + ".mp3"
        temp_trimmed_output_file = os.path.join(output_trimmed_folder1, output_filename)

        video_file_name = str(file.split(".")[0]) + ".mp4"
        video_file_path = os.path.join(video_folder1, video_file_name)
        start_time, end_time = find_timestamps_longest_scene(video_file_path)
        clean_audio_deepfilternet(temp_audio_file, temp_trimmed_wav_file, temp_trimmed_output_file, trim=True, start_time=start_time, end_time=end_time)

        mel_spectrogram_filename = "trimmed_mel_spectrogram_" + str(file.split(".")[0]) + ".npy"
        mfccs_filename = "trimmed_mfccs_" + str(file.split(".")[0]) + ".npy"
        temp_mel_spectrogram_filename = os.path.join(mel_spectrogram_folder, mel_spectrogram_filename)
        temp_mfccs_filename = os.path.join(mfccs_folder, mfccs_filename)
        i += 1

        S_db_temp, mfccs_temp = get_mel_spectrogram_and_mfccs(temp_trimmed_output_file, temp_mel_spectrogram_filename, temp_mfccs_filename)

else:
    temp_wav_file = "C:/Users/User/Downloads/temp_file.wav"
    output_folder1 = "C:/Users/User/OneDrive/Documents/Research Project Cleaned Audio Files/"
    mel_spectrogram_folder = "C:/Users/User/OneDrive/Documents/Research Project Audio Preprocessing/Mel Spectrograms/"
    mfccs_folder = "C:/Users/User/OneDrive/Documents/Research Project Audio Preprocessing/Mel Frequency Cepstral Coefficients/"
    i = 0
    for file in os.listdir(folder1):
        print(f"{i+1}) {file}")
        temp_audio_file = os.path.join(folder1, file)

        output_filename = "cleaned" + str(file.split(".")[0]) + ".mp3"
        temp_output_file = os.path.join(output_folder1, output_filename)
        clean_audio_deepfilternet(temp_audio_file, temp_wav_file, temp_output_file, trim=False, start_time=None, end_time=None)

        mel_spectrogram_filename = "mel_spectrogram_" + str(file.split(".")[0]) + ".npy"
        mfccs_filename = "mfccs_" + str(file.split(".")[0]) + ".npy"
        temp_mel_spectrogram_filename = os.path.join(mel_spectrogram_folder, mel_spectrogram_filename)
        temp_mfccs_filename = os.path.join(mfccs_folder, mfccs_filename)

        S_db_temp, mfccs_temp = get_mel_spectrogram_and_mfccs(temp_output_file, temp_mel_spectrogram_filename, temp_mfccs_filename)
        i += 1










