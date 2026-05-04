import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from barbar import Bar
import h5py
import librosa
import ast
import sys
import csv
from natsort import natsorted
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix
import os


def get_video_number(file_path):
    base_filename = os.path.basename(file_path)
    video_number = base_filename.split("eo")[1]
    video_number = video_number.split(".")[0]
    video_number = int(video_number)
    return video_number


def normalize_label(label, emotions=True):
    new_label = label / 100.0
    return new_label


def get_corresponding_data(video_number):
    # index = video_number - 1
    audio_embedding = audio_embeddings[video_number-1]
    text_embedding = text_embeddings[video_number-1]
    video_embedding = video_embeddings[video_number-1]
    new_label = new_labels_data[video_number-1]

    condition_label_file = "C:/Users/User/PycharmProjects/Research Project/Condition_Labels.csv"
    condition_label = pd.read_csv(condition_label_file)
    cond_label = condition_label['Neurodivergent'].loc[video_number-1]
    # Yes (1): Autism/Neurodivergent, No (0): Normal/Neurotypical

    # new_label = normalize_label(new_label, emotions=emotions)
    return audio_embedding, text_embedding, video_embedding, new_label, cond_label


def get_split_data(phase_split, phase_file_list):
    phase_split_audio_embeddings_list = []
    phase_split_text_embeddings_list = []
    phase_split_video_embeddings_list = []
    phase_split_label_list = []
    phase_split_condition_list = []
    for file in phase_file_list:
        video_number = get_video_number(file)
        audio_embedding, text_embedding, video_embedding, new_label, cond_label = get_corresponding_data(video_number)
        phase_split_audio_embeddings_list.append(audio_embedding)
        phase_split_text_embeddings_list.append(text_embedding)
        phase_split_video_embeddings_list.append(video_embedding)
        phase_split_label_list.append(new_label)
        phase_split_condition_list.append(cond_label)
    return phase_split_audio_embeddings_list, phase_split_text_embeddings_list, phase_split_video_embeddings_list, phase_split_label_list, phase_split_condition_list


def return_train_and_valid_indices(train_validation_split, epoch):
    remainder = epoch % 5
    split_size = int(len(train_validation_split) / 5)
    indices = np.arange(len(train_validation_split))
    valid_indices = indices[(remainder*(split_size)):((remainder+1)*split_size)]
    train_indices = np.concatenate([indices[:(remainder*split_size)], indices[((remainder+1)*split_size):]])
    return train_indices, valid_indices


class MultimodalDataset(Dataset):
    def __init__(self, audio_embeddings, text_embeddings, video_embeddings, labels, cond_labels):
        super().__init__()
        self.audio_embeddings = audio_embeddings
        self.text_embeddings = text_embeddings
        self.video_embeddings = video_embeddings
        self.labels = labels
        self.cond_labels = cond_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_embedding = torch.tensor(self.audio_embeddings[idx], dtype=torch.float32).unsqueeze(0)
        text_embedding = torch.tensor(self.text_embeddings[idx], dtype=torch.float32).unsqueeze(0)
        video_embedding = torch.tensor(self.video_embeddings[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        combined_embedding = torch.cat([audio_embedding, text_embedding, video_embedding], dim=1)
        cond_label = self.cond_labels[idx]
        return combined_embedding, label, cond_label


class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# emotions = True
audio_embeddings_file = "C:/Users/User/PycharmProjects/Research Project/audio_embeddings_pretrained_emotions_mlc_wav2vec2.npy"
text_embeddings_file = "C:/Users/User/PycharmProjects/Research Project/text_embeddings_pretrained_emotions_mlc.npy"
video_embeddings_file = "C:/Users/User/PycharmProjects/Research Project/video_embeddings_pretrained_emotions_mlc.npy"
labels_file = "C:/Users/User/PycharmProjects/Research Project/New_Labels_By_Classification_Emotions_Threshold15.csv"

audio_embeddings = np.load(audio_embeddings_file)
text_embeddings = np.load(text_embeddings_file)
video_embeddings = np.load(video_embeddings_file)
labels_data = np.genfromtxt(labels_file, delimiter=',')
new_labels_data = [arr[:7] for arr in labels_data]
new_labels_data = np.array(new_labels_data)

print(f"audio_embeddings shape: {audio_embeddings.shape}")
print(f"text_embeddings shape: {text_embeddings.shape}")
print(f"video_embeddings shape: {video_embeddings.shape}")
print(f"labels_data shape: {new_labels_data.shape}")


# index = video_number - 1
train_validation_split_file = "C:/Users/User/PycharmProjects/Research Project/Train_Validation_Split.csv"
train_validation_split = np.loadtxt(train_validation_split_file, delimiter=',', dtype=str)
train_validation_split = train_validation_split.tolist()
train_validation_split_audio_embeddings_list, train_validation_split_text_embeddings_list, train_validation_split_video_embeddings_list, train_validation_split_label_list, train_validation_split_condition_list = get_split_data("Train-Validation", train_validation_split)
train_validation_dataset = MultimodalDataset(train_validation_split_audio_embeddings_list, train_validation_split_text_embeddings_list, train_validation_split_video_embeddings_list, train_validation_split_label_list, train_validation_split_condition_list)

model = MultimodalModel()
# criterion = nn.MSELoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
num_epochs = 30


for epoch in range(num_epochs):
    model.train()

    train_indices, valid_indices = return_train_and_valid_indices(train_validation_split, epoch)
    train_dataset = Subset(train_validation_dataset, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = Subset(train_validation_dataset, valid_indices)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Training
    total_train_loss = 0.0
    train_pred_array = []
    train_label_array = []
    for combined_embedding, label, cond_label in Bar(train_loader):
        output = model(combined_embedding)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prob = torch.sigmoid(output)
        pred = (prob > 0.35).float()
        train_pred_array.append(pred.detach())
        train_label_array.append(label.detach())

        total_train_loss += loss.item()

    train_pred_temp = torch.cat(train_pred_array, dim=0)
    train_label_temp = torch.cat(train_label_array, dim=0)
    train_preds_array = train_pred_temp.view(-1).detach().cpu().numpy()
    train_labels_array = train_label_temp.view(-1).detach().cpu().numpy()

    train_accuracy = accuracy_score(train_labels_array, train_preds_array)
    train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Train Accuracy: {train_accuracy:.5f}, Train Loss: {train_loss:.6f}")

    # Validation
    model.eval()
    total_val_loss = 0.0
    val_pred_array = []
    val_label_array = []
    with torch.no_grad():
        for combined_embedding, label, cond_label in Bar(val_loader):
            output = model(combined_embedding)
            loss = criterion(output, label)

            prob = torch.sigmoid(output)
            pred = (prob > 0.35).float()
            val_pred_array.append(pred.detach())
            val_label_array.append(label.detach())

            total_val_loss += loss.item()

    val_pred_temp = torch.cat(val_pred_array, dim=0)
    val_label_temp = torch.cat(val_label_array, dim=0)
    val_preds_array = val_pred_temp.view(-1).detach().cpu().numpy()
    val_labels_array = val_label_temp.view(-1).detach().cpu().numpy()

    val_accuracy = accuracy_score(val_labels_array, val_preds_array)
    val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}: Validation Accuracy: {val_accuracy:.5f}, Validation Loss: {val_loss:.6f}")

    if epoch == (num_epochs - 1):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss
        }, f"multimodal_model_weights_emotions_mlc.pth")


# Load Model
weights_file = f"multimodal_model_weights_emotions_mlc.pth"
checkpoint = torch.load(weights_file, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])

test_split_file = "C:/Users/User/PycharmProjects/Research Project/Test_Split.csv"
test_split = np.loadtxt(test_split_file, delimiter=',', dtype=str)
test_split = test_split.tolist()
test_split_audio_embeddings_list, test_split_text_embeddings_list, test_split_video_embeddings_list, test_split_label_list, test_split_condition_list = get_split_data("Test", test_split)
test_dataset = MultimodalDataset(test_split_audio_embeddings_list, test_split_text_embeddings_list, test_split_video_embeddings_list, test_split_label_list, test_split_condition_list)
test_loader = DataLoader(test_dataset, batch_size=16)

model.eval()
total_test_loss = 0.0
test_pred_array = []
test_label_array = []
neurotypical_pred_array = []
neurotypical_label_array = []
neurodivergent_pred_array = []
neurodivergent_label_array = []
with torch.no_grad():
    for combined_embedding, label, cond_label in Bar(test_loader):
        output = model(combined_embedding)
        loss = criterion(output, label)

        prob = torch.sigmoid(output)
        pred = (prob > 0.35).float()
        test_pred_array.append(pred.detach())
        test_label_array.append(label.detach())

        for i in range(cond_label.shape[0]):
            if cond_label[i] == 1:
                neurodivergent_pred_array.append(pred[i].detach())
                neurodivergent_label_array.append(label[i].detach())
            elif cond_label[i] == 0:
                neurotypical_pred_array.append(pred[i].detach())
                neurotypical_label_array.append(label[i].detach())

        total_test_loss += loss.item()

test_pred_temp = torch.cat(test_pred_array, dim=0)
test_label_temp = torch.cat(test_label_array, dim=0)
test_preds_array = test_pred_temp.view(-1).detach().cpu().numpy()
test_labels_array = test_label_temp.view(-1).detach().cpu().numpy()

label_emotion_names = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
label_attribute_names = ["Valence", "Arousal", "Dominance"]

test_accuracy = accuracy_score(test_labels_array, test_preds_array)
f1_micro = f1_score(test_labels_array, test_preds_array, average='micro')
f1_macro = f1_score(test_labels_array, test_preds_array, average='macro')
f1_weighted = f1_score(test_labels_array, test_preds_array, average='weighted')
f1_per_label = f1_score(test_label_temp.squeeze(1).detach().cpu().numpy(), test_pred_temp.squeeze(1).detach().cpu().numpy(), average=None)
test_loss = total_test_loss / len(test_loader)

print(f"Multimodal Predictions: {pred}")
print(f"Multimodal Labels: {label}")
print(f"Multimodal Accuracy: {test_accuracy:.5f}")
print(f"Multimodal Loss: {test_loss:.6f}")
print(f"Multimodal F1_Micro: {f1_micro:.5f}")
print(f"Multimodal F1_Macro: {f1_macro:.5f}")
print(f"Multimodal F1_Weighted: {f1_weighted:.5f}")
print(f"Multimodal F1 Per Emotion:")
for label_name, f in zip(label_emotion_names, f1_per_label):
    print(f"{label_name}: {f:.5f}")



# Evaluate Neurotypical
neurotypical_pred_temp = torch.cat(neurotypical_pred_array, dim=0)
neurotypical_label_temp = torch.cat(neurotypical_label_array, dim=0)
neurotypical_preds_array = neurotypical_pred_temp.view(-1).detach().cpu().numpy()
neurotypical_labels_array = neurotypical_label_temp.view(-1).detach().cpu().numpy()

accuracy_neurotypical = accuracy_score(neurotypical_labels_array, neurotypical_preds_array)
f1_micro_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='micro')
f1_macro_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='macro')
f1_weighted_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='weighted')
neurotypical_labels = np.array([x.detach().cpu().numpy() for x in neurotypical_label_array]).squeeze(1)
neurotypical_preds = np.array([x.detach().cpu().numpy() for x in neurotypical_pred_array]).squeeze(1)
print(neurotypical_labels.shape)
print(neurotypical_preds.shape)
f1_per_label_neurotypical = f1_score(neurotypical_labels, neurotypical_preds, average=None)

print(f"\nNeurotypical")
print(f"Accuracy: {accuracy_neurotypical:.6f}")
print(f"F1_Micro: {f1_micro_neurotypical:.5f}")
print(f"F1_Macro: {f1_macro_neurotypical:.5f}")
print(f"F1_Weighted: {f1_weighted_neurotypical:.5f}")
print(f"F1 Per Emotion for Neurotypical:")
for l, f in zip(label_emotion_names, f1_per_label_neurotypical):
    print(f"{l}: {f:.5f}")



# Evaluate Neurodivergent
neurodivergent_pred_temp = torch.cat(neurodivergent_pred_array, dim=0)
neurodivergent_label_temp = torch.cat(neurodivergent_label_array, dim=0)
neurodivergent_preds_array = neurodivergent_pred_temp.view(-1).detach().cpu().numpy()
neurodivergent_labels_array = neurodivergent_label_temp.view(-1).detach().cpu().numpy()

accuracy_neurodivergent = accuracy_score(neurodivergent_labels_array, neurodivergent_preds_array)
f1_micro_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='micro')
f1_macro_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='macro')
f1_weighted_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='weighted')
neurodivergent_labels = np.array([x.detach().cpu().numpy() for x in neurodivergent_label_array]).squeeze(1)
neurodivergent_preds = np.array([x.detach().cpu().numpy() for x in neurodivergent_pred_array]).squeeze(1)
f1_per_label_neurodivergent = f1_score(neurodivergent_labels, neurodivergent_preds, average=None)

print(f"\nNeurodivergent")
print(f"Accuracy: {accuracy_neurodivergent:.6f}")
print(f"F1_Micro: {f1_micro_neurodivergent:.5f}")
print(f"F1_Macro: {f1_macro_neurodivergent:.5f}")
print(f"F1_Weighted: {f1_weighted_neurodivergent:.5f}")
print(f"F1 Per Emotion for Neurodivergent:")
for l, f in zip(label_emotion_names, f1_per_label_neurodivergent):
    print(f"{l}: {f:.5f}")












