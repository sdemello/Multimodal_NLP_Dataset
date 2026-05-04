import numpy as np
import pandas as pd
import os
from pathlib import Path
from natsort import natsorted
import random
import csv

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
random.seed(42)

folder1 = "C:/Users/User/OneDrive/Documents/ResearchProjectHDF5Files/"
files_list = os.listdir(folder1)

def get_files(file_list, condition):
    dataset_file = "C:/Users/User/Downloads/Research Project Dataset.csv"
    dataset_table = pd.read_csv(dataset_file, index_col=0)
    if condition == "Normal":
        files_dataset = dataset_table[dataset_table["Neurodivergent"] == "No"]
    elif condition == "Neurodivergent":
        files_dataset = dataset_table[dataset_table["Neurodivergent"] == "Yes"]
    files_clip_numbers = files_dataset.index.to_numpy()

    files = []
    for file in file_list:
        string1 = file.split("output_Video")
        file_number = int(string1[1].split(".")[0])
        if file_number in files_clip_numbers:
            temp_file = os.path.join(folder1, file)
            files.append(temp_file)

    return files

normal_files = get_files(files_list, "Normal")
neurodivergent_files = get_files(files_list, "Neurodivergent")

def get_randomized_split_list(files_list, split_percent):
    shuffled_list = files_list[:]
    random.shuffle(shuffled_list)
    split_point = round(len(files_list)*(split_percent/100))
    first_list = shuffled_list[:split_point]
    second_list = shuffled_list[split_point:]
    return first_list, second_list

# Train and Validation Split (75%): 555 Normal, 195 Neurodivergent; Test Split (25%): 185 Normal, 65 Neurodivergent
trainvalidation_and_test_split = 75
normal_train_and_validation_split, normal_test_split = get_randomized_split_list(normal_files, trainvalidation_and_test_split)
neurodivergent_train_and_validation_split, neurodivergent_test_split = get_randomized_split_list(neurodivergent_files, trainvalidation_and_test_split)

# Train Split (60%): 444 Normal, 156 Neurodivergent; Validation Split (15%): 111 Normal, 39 Neurodivergent
train_and_validation_split = 80
normal_train_split, normal_validation_split = get_randomized_split_list(normal_train_and_validation_split, train_and_validation_split)
neurodivergent_train_split, neurodivergent_validation_split = get_randomized_split_list(neurodivergent_train_and_validation_split, train_and_validation_split)

# Combine Normal and Neurodivergent Splits and Shuffle
train_split = normal_train_split + neurodivergent_train_split
random.shuffle(train_split)
validation_split = normal_validation_split + neurodivergent_validation_split
random.shuffle(validation_split)
test_split = normal_test_split + neurodivergent_test_split
random.shuffle(test_split)

# Export to csv files
splits = [train_split, validation_split, test_split]
output_files = ["Train_Split.csv", "Validation_Split.csv", "Test_Split.csv"]
for i in range(len(splits)):
    temp_output_filename = output_files[i]
    temp_output_file_directory = os.path.join("C:/Users/User/Downloads", temp_output_filename)
    with open(temp_output_file_directory, 'w', newline='') as f:
        writer = csv.writer(f)
        for item in splits[i]:
            writer.writerow([item])









