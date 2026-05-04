import numpy as np
import os
import cv2
import math
from moviepy.editor import VideoFileClip, ImageSequenceClip
import moviepy.editor as mp
import pandas as pd
import h5py
import sys
from Find_Timestamps_Scene_Changes import *


def frame_extraction_sample_clips(video_clip, num_samples):
    # Generate 16 evenly spaced timestamps
    timestamps = np.linspace(0, video_clip.duration, num_samples)

    frames = []
    for t in timestamps:
        frame = video_clip.get_frame(t)
        frames.append(frame)

    # Create new video from sampled frames
    frame_extracted_clip = ImageSequenceClip(frames, fps=2)

    return frame_extracted_clip


def video_preprocess(video_file, output_folder, trim, perform_frame_extraction):
    path1 = Path(video_file)
    video_name = path1.name
    video_base_name = video_name.split(".")[0]
    video_clip = VideoFileClip(video_file)
    width = video_clip.w; height = video_clip.h
    print(f"Resolution: {width}, {height}")

    if width > height:
        diff = width - height
        left = 0; right = 0; top = int(round(diff/2)); bottom = int(round(diff/2)); color = (0, 0, 0)
    elif height > width:
        diff = height - width
        left = int(round(diff/2)); right = int(round(diff/2)); top = 0; bottom = 0; color = (0, 0, 0)

    if perform_frame_extraction is True:
        video_clip = frame_extraction_sample_clips(video_clip, num_samples=16)
        print("Frame Extraction Completed")

    padded_clip = video_clip.margin(top, bottom, left, right, color)
    print("Padded Clip Completed")

    if trim is True:
        start_time, end_time = find_timestamps_longest_scene(video_file)
        trimmed_clip = padded_clip.subclip(start_time, end_time)
        print("Trimmed Clip Completed")

        resized_clip = trimmed_clip.resize(newsize=(224, 224))
        print("Resized Clip Completed")
    else:
        resized_clip = padded_clip.resize(newsize=(224, 224))
        print("Resized Clip Completed")

    def video_to_hdf5(clip, hdf5_path):
        frames = []
        # Normalization per frame
        for frame in clip.iter_frames():
            normalized_frame = frame.astype(np.float32) / 255.0
            frames.append(normalized_frame)
        # Convert list to numpy array (frames, height, width, channels)
        video_data = np.stack(frames)
        # Save to HDF5
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('video_data', data=video_data, compression="gzip")
            f.attrs['fps'] = clip.fps
        return video_data

    output_file_name = "output_" + video_base_name + ".h5"
    hdf5_path = os.path.join(output_folder, output_file_name)
    hdf5_video_data = video_to_hdf5(resized_clip, hdf5_path)
    print(f"Output saved to {hdf5_path}")

    video_clip.close(); padded_clip.close(); resized_clip.close()


trim = False
perform_frame_extraction = True
folder1 = "C:/Users/User/OneDrive/Documents/Research Project Data - Copy/"
if trim is True:
    output_folder = "C:/Users/User/OneDrive/Documents/Output_Preprocessed_Trimmed_Video_Clips/"
elif trim is False and perform_frame_extraction is False:
    output_folder = "C:/Users/User/OneDrive/Documents/Output_Preprocessed_Video_Clips/"
elif trim is False and perform_frame_extraction is True:
    output_folder = "C:/Users/User/OneDrive/Documents/Output_Preprocessed_Video_Clips_Frame_Extraction/"
i = 0
for file in os.listdir(folder1):
    print(f"{i+1}) {file}")
    temp_video_file = os.path.join(folder1, file)
    video_preprocess(temp_video_file, output_folder, trim, perform_frame_extraction)
    i += 1









