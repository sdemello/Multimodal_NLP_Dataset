import numpy as np
import pandas as pd
import os
import ast

labels_file = "C:/Users/User/PycharmProjects/Research Project/Labels_File.csv"
labels_data = np.genfromtxt(labels_file, delimiter=',')
emotions = True
if emotions is True:
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

export_file_csv = f"C:/Users/User/Downloads/New_Labels_By_Classification_Emotions_Threshold{str(threshold)}.csv"
# np.savetxt(export_file_csv, new_revised_temp_label_table, delimiter=',', fmt='%d')

export_file_npy = f"C:/Users/User/Downloads/New_Labels_By_Classification_Emotions_Threshold{str(threshold)}.npy"
# np.save(export_file_npy, new_revised_temp_label_table)







