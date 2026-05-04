import numpy as np
import pandas as pd
import ast
import os
import matplotlib.pyplot as plt


# Make sure you include more analysis on the dataset itself
# label frequency distribution
# inter-annotator agreement
# per-emotion breakdown
# commonly confused emotion pairs.
#
# Figures to prioritize, in order:
# dataset composition (label distribution, duration, percentage of videos with folks on-the-spectrum (this is your key contribution))
# data collection and annotation pipeline diagram
# one qualitative example with multimodal features and labels
# a per-emotion bar chart comparing unimodal versus fusion if space allows

labels_file = "C:/Users/User/PycharmProjects/Research Project/Labels_File.csv"
labels_data = np.genfromtxt(labels_file, delimiter=',')
choose_emotions = True
emotions = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
attributes = ["Valence", "Arousal", "Dominance"]
if choose_emotions is True:
    new_labels_data = [arr[:7] for arr in labels_data]
else:
    new_labels_data = [arr[7:] for arr in labels_data]
new_labels_data = np.array(new_labels_data)
print(f"labels_data shape: {new_labels_data.shape}")
print(new_labels_data)

new_revised_temp_label_table = []
threshold = 15
for temp_label in new_labels_data:
    new_revised_temp_label_indices = (temp_label > threshold).astype(int)
    new_revised_temp_label_table.append(new_revised_temp_label_indices)
print(new_revised_temp_label_table)
new_revised_temp_label_table_np = np.array(new_revised_temp_label_table)

most_prominent_emotion = np.zeros((7,), dtype=np.int32)
print(most_prominent_emotion)
for temp_label in new_labels_data:
    indices = np.argmax(temp_label)
    most_prominent_emotion[indices] += 1
print(emotions)
print(most_prominent_emotion)
plt.bar(emotions, most_prominent_emotion)
plt.show()


def get_emotion_count(labels_data):
    emotion_anger_count = 0
    emotion_disgust_count = 0
    emotion_fear_count = 0
    emotion_happy_count = 0
    emotion_sad_count = 0
    emotion_surprise_count = 0
    emotion_neutral_count = 0

    for i in range(labels_data.shape[0]):
        temp_label = labels_data[i]
        emotion_anger_count += temp_label[0]
        emotion_disgust_count += temp_label[1]
        emotion_fear_count += temp_label[2]
        emotion_happy_count += temp_label[3]
        emotion_sad_count += temp_label[4]
        emotion_surprise_count += temp_label[5]
        emotion_neutral_count += temp_label[6]

    print(emotion_anger_count)
    print(emotion_disgust_count)
    print(emotion_fear_count)
    print(emotion_happy_count)
    print(emotion_sad_count)
    print(emotion_surprise_count)
    print(emotion_neutral_count)


print("\n\nLabels past Threshold of 15: ")
get_emotion_count(new_revised_temp_label_table_np)

"""
clip_lengths = []
from moviepy.editor import VideoFileClip
folder_path_clips = "C:/Users/User/OneDrive/Documents/Research Project Data - Copy/"
for file in os.listdir(folder_path_clips):
    filepath = os.path.join(folder_path_clips, file)
    with VideoFileClip(filepath) as clip:
        print(f"{file}, {clip.duration}")
        clip_lengths.append(clip.duration)
print(clip_lengths)
import csv
export_file = "C:/Users/User/Downloads/Clip_Lengths.csv"
# with open(export_file, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(clip_lengths)
"""


data_ratings_file = "C:/Users/User/Downloads/Ratings Data.csv"
dataset_all_ratings = pd.read_csv(data_ratings_file)
import krippendorff
for emotion in emotions:
    temp_df = dataset_all_ratings[["Filename", "Email", emotion]].copy()
    temp_df.columns = ['clip_id', 'rater_id', 'rating']
    temp_df['rating'] = pd.to_numeric(temp_df['rating'], errors='coerce')
    pivot = temp_df.pivot_table(index='rater_id', columns='clip_id', values='rating')
    data = pivot.to_numpy()
    krippendorff_alpha = krippendorff.alpha(reliability_data=data, level_of_measurement='interval')
    print(f"{emotion}: {krippendorff_alpha:.3f}")


import csv
export_file = "C:/Users/User/Downloads/Clip_Lengths.csv"
clip_lengths = []
with open(export_file, "r", newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        clip_lengths.append(row)
clip_lengths = clip_lengths[0]
clip_lengths = np.array(clip_lengths)
clip_lengths = [float(x) for x in clip_lengths]
print(clip_lengths)
sum_clip_lengths = np.sum(clip_lengths)
print(sum_clip_lengths)







