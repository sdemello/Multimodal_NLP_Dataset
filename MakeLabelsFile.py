import numpy as np
import pandas as pd
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

labels_file = "C:/Users/User/Downloads/Ratings Data - Sheet1.csv"
labels_pd = pd.read_csv(labels_file)
# print(labels_pd)

new_labels_pd = labels_pd.sort_values(by="Filename", ascending=True).reset_index(drop=True)
# print(new_labels_pd)

anger_label = new_labels_pd["Anger"]
disgust_label = new_labels_pd["Disgust"]
fear_label = new_labels_pd["Fear"]
happy_label = new_labels_pd["Happy"]
sad_label = new_labels_pd["Sad"]
surprise_label = new_labels_pd["Surprise"]
neutral_label = new_labels_pd["Neutral"]
valence_label = new_labels_pd["Valence"]
arousal_label = new_labels_pd["Arousal"]
domination_label = new_labels_pd["Domination"]

table_list = []
for i in range(1, 1001):
    filename = "Video" + str(i) + ".mp4"
    filename_count = (new_labels_pd["Filename"] == filename).sum()
    indices = new_labels_pd.index[new_labels_pd["Filename"] == filename]
    indices = indices.tolist()
    print(f"{filename}: {filename_count}, {indices}")

    # Anger Emotion
    temp_sum_anger = 0.0
    anger_list = []
    for index in indices:
        anger_list.append(anger_label.loc[index])
        temp_sum_anger += anger_label.loc[index]
    temp_avg_anger = round((temp_sum_anger / len(indices)), 1)

    # Disgust Emotion
    temp_sum_disgust = 0.0
    disgust_list = []
    for index in indices:
        disgust_list.append(disgust_label.loc[index])
        temp_sum_disgust += disgust_label.loc[index]
    temp_avg_disgust = round((temp_sum_disgust / len(indices)), 1)

    # Fear Emotion
    temp_sum_fear = 0.0
    fear_list = []
    for index in indices:
        fear_list.append(fear_label.loc[index])
        temp_sum_fear += fear_label.loc[index]
    temp_avg_fear = round((temp_sum_fear / len(indices)), 1)

    # Happy Emotion
    temp_sum_happy = 0.0
    happy_list = []
    for index in indices:
        happy_list.append(happy_label.loc[index])
        temp_sum_happy += happy_label.loc[index]
    temp_avg_happy = round((temp_sum_happy / len(indices)), 1)

    # Sad Emotion
    temp_sum_sad = 0.0
    sad_list = []
    for index in indices:
        sad_list.append(sad_label.loc[index])
        temp_sum_sad += sad_label.loc[index]
    temp_avg_sad = round((temp_sum_sad / len(indices)), 1)

    # Surprise Emotion
    temp_sum_surprise = 0.0
    surprise_list = []
    for index in indices:
        surprise_list.append(surprise_label.loc[index])
        temp_sum_surprise += surprise_label.loc[index]
    temp_avg_surprise = round((temp_sum_surprise / len(indices)), 1)

    # Neutral Emotion
    temp_sum_neutral = 0.0
    neutral_list = []
    for index in indices:
        neutral_list.append(neutral_label.loc[index])
        temp_sum_neutral += neutral_label.loc[index]
    temp_avg_neutral = round((temp_sum_neutral / len(indices)), 1)

    # Valence Attribute
    temp_sum_valence = 0.0
    valence_list = []
    for index in indices:
        valence_list.append(valence_label.loc[index])
        temp_sum_valence += valence_label.loc[index]
    temp_avg_valence = round((temp_sum_valence / len(indices)), 2)

    # Arousal Attribute
    temp_sum_arousal = 0.0
    arousal_list = []
    for index in indices:
        arousal_list.append(arousal_label.loc[index])
        temp_sum_arousal += arousal_label.loc[index]
    temp_avg_arousal = round((temp_sum_arousal / len(indices)), 2)

    # Domination Attribute
    temp_sum_domination = 0.0
    domination_list = []
    for index in indices:
        domination_list.append(domination_label.loc[index])
        temp_sum_domination += domination_label.loc[index]
    temp_avg_domination = round((temp_sum_domination / len(indices)), 2)

    table_row = [filename, temp_avg_anger, temp_avg_disgust, temp_avg_fear, temp_avg_happy, temp_avg_sad, temp_avg_surprise, temp_avg_neutral, temp_avg_valence, temp_avg_arousal, temp_avg_domination]
    table_list.append(table_row)

table_pd = pd.DataFrame(data=table_list, columns=["Filename", "Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral", "Valence", "Arousal", "Domination"])
print(table_pd)
file_to_export = "C:/Users/User/Downloads/ResearchProjectLabels.csv"
table_pd.to_csv(file_to_export, index=False)








