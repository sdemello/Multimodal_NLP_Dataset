import numpy as np
import pandas as pd
import os
import librosa
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from barbar import Bar
import sys
import csv
from itertools import chain
import torchvision
from transformers import VivitImageProcessor, VivitModel, VivitConfig
from natsort import natsorted
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix
np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)


def get_video_number(file_path):
    file = os.path.basename(file_path)
    video_number = file.split("eo")[1]
    video_number = int(video_number.split(".")[0])
    return video_number


def normalize_label(label):
    new_label = new_label / 100.0
    # new_label[number] = (new_label[number] + 3) / 6
    return new_label


def get_corresponding_data(video_number):
    hdf5_folder = r"C:/Users/User/OneDrive/Documents/ResearchProjectHDF5Files/"
    hdf5_filename = "output_Video" + str(video_number) + ".h5"
    hdf5_file = os.path.join(hdf5_folder, hdf5_filename)
    with h5py.File(hdf5_file, 'r') as f:
        video_data = f['video_data']
        label_data = f['label']
        video_data = video_data[:]

    labels_file = "C:/Users/User/PycharmProjects/Research Project/New_Labels_By_Classification_Emotions_Threshold15.npy"
    labels_data = np.load(labels_file)
    label_clip = labels_data[video_number - 1]
    label_clip = label_clip.astype(float)

    condition_label_file = "C:/Users/User/PycharmProjects/Research Project/Condition_Labels.csv"
    condition_label = pd.read_csv(condition_label_file)
    cond_label = condition_label['Neurodivergent'].loc[video_number - 1]
    # Yes (1): Autism/Neurodivergent, No (0): Normal/Neurotypical

    return video_data, label_clip, cond_label


def make_whole_dataset():
    whole_vid_data_list = []
    whole_label_list = []
    whole_condition_list = []
    for i in range(1000):
        vid_data, label_clip, cond_label = get_corresponding_data(i + 1)
        whole_vid_data_list.append(vid_data)
        whole_label_list.append(label_clip)
        whole_condition_list.append(cond_label)
    return whole_vid_data_list, whole_label_list, whole_condition_list


def get_split_data(phase_split, phase_file_list):
    phase_split_vid_data_list = []
    phase_split_label_list = []
    phase_split_condition_list = []
    for file in phase_file_list:
        video_number = get_video_number(file)
        vid_data, new_label_clip, cond_label = get_corresponding_data(video_number)
        phase_split_vid_data_list.append(vid_data)
        phase_split_label_list.append(new_label_clip)
        phase_split_condition_list.append(cond_label)
    return phase_split_vid_data_list, phase_split_label_list, phase_split_condition_list


def return_train_and_valid_indices(train_validation_split, epoch):
    remainder = epoch % 5
    split_size = int(len(train_validation_split) / 5)
    indices = np.arange(len(train_validation_split))
    valid_indices = indices[(remainder*(split_size)):((remainder+1)*split_size)]
    train_indices = np.concatenate([indices[:(remainder*split_size)], indices[((remainder+1)*split_size):]])
    return train_indices, valid_indices


class VideoDataset(Dataset):
    def __init__(self, video_data, labels, cond_labels):
        self.video_data = video_data
        self.labels = labels
        self.cond_labels = cond_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_clip_data = torch.from_numpy(self.video_data[idx])
        video_clip_data = video_clip_data.permute(0, 3, 1, 2)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        cond_label = self.cond_labels[idx]
        return video_clip_data, label, cond_label


# Get Train-Validation Split
train_validation_split_file = "C:/Users/User/Downloads/Train_Validation_Split.csv"
train_validation_split = np.loadtxt(train_validation_split_file, delimiter=',', dtype=str)
train_validation_split = train_validation_split.tolist()
train_validation_vid_data_list, train_validation_label_list, train_validation_condition_list = get_split_data("Train-Validation", train_validation_split)
train_validation_dataset = VideoDataset(train_validation_vid_data_list, train_validation_label_list, train_validation_condition_list)

# Configure Model to 16 Frames
config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
config.num_frames = 16
model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", config=config, ignore_mismatched_sizes=True)

# Freeze most of backbone
for param in model.parameters():
    param.requires_grad = False

# criterion = nn.SmoothL1Loss()
criterion = nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
head = torch.nn.Linear(768, 7)
head = head.to(device)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(head.parameters()),
    lr=1e-4
)
# num_epochs = 20
num_epochs = 5


for epoch in range(num_epochs):
    # Training
    model.train()
    print(f"Epoch: {epoch+1}")

    train_indices, valid_indices = return_train_and_valid_indices(train_validation_dataset, epoch)
    train_dataset = Subset(train_validation_dataset, train_indices)
    valid_dataset = Subset(train_validation_dataset, valid_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=4)

    total_train_loss = 0.0
    train_pred_array = []
    train_label_array = []
    for video_data, label, cond_label in Bar(train_dataloader):
        video_data = video_data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(video_data)
        cls = output.last_hidden_state.mean(dim=1)
        pred = head(cls)
        loss = criterion(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        prob = torch.sigmoid(pred)
        pred = (prob > 0.35).float()
        train_pred_array.append(pred.detach())
        train_label_array.append(label.detach())

        total_train_loss += loss.item()

    train_pred_temp = torch.cat(train_pred_array, dim=0)
    train_label_temp = torch.cat(train_label_array, dim=0)
    train_preds_array = train_pred_temp.view(-1).numpy()
    train_labels_array = train_label_temp.view(-1).numpy()

    train_accuracy = accuracy_score(train_labels_array, train_preds_array)
    train_loss = total_train_loss / len(train_dataloader)
    print(f"Video Epoch {epoch+1}: Train Accuracy: {train_accuracy:.5f}, Train Loss = {train_loss:.5f}")

    # Validation
    model.eval()
    total_val_loss = 0.0
    val_pred_array = []
    val_label_array = []
    with torch.no_grad():
        for video_data, label, cond_label in Bar(valid_dataloader):
            video_data = video_data.to(device)
            label = label.to(device)

            output = model(video_data)
            cls = output.last_hidden_state.mean(dim=1)
            pred = head(cls)
            loss = criterion(pred, label)

            prob = torch.sigmoid(pred)
            pred = (prob > 0.35).float()
            val_pred_array.append(pred.detach())
            val_label_array.append(label.detach())

            total_val_loss += loss.item()

        val_pred_temp = torch.cat(val_pred_array, dim=0)
        val_label_temp = torch.cat(val_label_array, dim=0)
        val_preds_array = val_pred_temp.view(-1).detach().cpu().numpy()
        val_labels_array = val_label_temp.view(-1).detach().cpu().numpy()

        val_accuracy = accuracy_score(val_labels_array, val_preds_array)
        val_loss = total_val_loss / len(valid_dataloader)
        print(f"Video Epoch {epoch+1}: Val Accuracy: {val_accuracy:.5f}, Validation Loss: {val_loss:.5f}")

    if epoch == (num_epochs - 1):
        torch.save({
            'model_state_dict': model.state_dict(),
            'head_state_dict': head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss
        }, f"video_weights_epoch{num_epochs}_emotions_mlc.pth")

# Load Model for Evaluation
weights_file = f"video_weights_epoch{num_epochs}_emotions_mlc.pth"
checkpoint = torch.load(weights_file)
model.load_state_dict(checkpoint['model_state_dict'])
head.load_state_dict(checkpoint['head_state_dict'])

# Get Test Split
test_split_file = "C:/Users/User/Downloads/Test_Split.csv"
test_split = np.loadtxt(test_split_file, delimiter=',', dtype=str)
test_split = test_split.tolist()
test_split_vid_data_list, test_split_label_list, test_split_condition_list = get_split_data("Test", test_split)

test_dataset = VideoDataset(test_split_vid_data_list, test_split_label_list, test_split_condition_list)
test_dataloader = DataLoader(test_dataset, batch_size=4)

# Testing
model.eval()
total_test_loss = 0.0
test_pred_array = []
test_label_array = []
neurotypical_pred_array = []
neurotypical_label_array = []
neurodivergent_pred_array = []
neurodivergent_label_array = []
with torch.no_grad():
    for video_data, label, cond_label in Bar(test_dataloader):
        video_data = video_data.to(device)
        label = label.to(device)

        output = model(video_data)
        cls = output.last_hidden_state.mean(dim=1)
        pred = head(cls)
        loss = criterion(pred, label)

        prob = torch.sigmoid(pred)
        pred = (prob > 0.35).float()
        test_pred_array.append(pred.detach())
        test_label_array.append(label.detach())

        for i in range(cond_label.shape[0]):
            if cond_label[i].item() == 1:
                neurodivergent_pred_array.append(pred[i].detach())
                neurodivergent_label_array.append(label[i].detach())
            elif cond_label[i].item() == 0:
                neurotypical_pred_array.append(pred[i].detach())
                neurotypical_label_array.append(label[i].detach())

        total_test_loss += loss.item()

test_pred_temp = torch.cat(test_pred_array, dim=0)
test_label_temp = torch.cat(test_label_array, dim=0)
test_preds_array = test_pred_temp.view(-1).detach().cpu().numpy()
test_labels_array = test_label_temp.view(-1).detach().cpu().numpy()

test_accuracy = accuracy_score(test_labels_array, test_preds_array)
test_f1_micro = f1_score(test_labels_array, test_preds_array, average='micro')
test_f1_macro = f1_score(test_labels_array, test_preds_array, average='macro')
test_f1_weighted = f1_score(test_labels_array, test_preds_array, average='weighted')
test_f1_per_label = f1_score(test_label_temp.squeeze(1).detach().cpu().numpy(), test_pred_temp.squeeze(1).detach().cpu().numpy(), average=None)
test_loss = total_test_loss / len(test_dataloader)

label_emotion_names = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
label_attribute_names = ["Valence", "Arousal", "Dominance"]

print(f"Video Test Predictions: {pred}")
print(f"Video Test Labels: {label}")
print(f"Video Test Accuracy: {test_accuracy:.5f}")
print(f"Video Test Loss: {test_loss:.6f}")
print(f"Video Test F1_Micro: {test_f1_micro:.5f}")
print(f"Video Test F1_Macro: {test_f1_macro:.5f}")
print(f"Video Test F1_Weighted: {test_f1_weighted:.5f}")
print(f"Video Test F1 Per Emotion:")
for l, f in zip(label_emotion_names, test_f1_per_label):
    print(f"{l}: {f:.5f}")


# Test Neurotypical Metrics Only
neurotypical_pred_temp = torch.cat(neurotypical_pred_array, dim=0)
neurotypical_label_temp = torch.cat(neurotypical_label_array, dim=0)
neurotypical_preds_array = neurotypical_pred_temp.view(-1).detach().cpu().numpy()
neurotypical_labels_array = neurotypical_label_temp.view(-1).detach().cpu().numpy()

accuracy_neurotypical = accuracy_score(neurotypical_labels_array, neurotypical_preds_array)
f1_micro_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='micro')
f1_macro_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='macro')
f1_weighted_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='weighted')
neurotypical_labels = np.array([x.detach().cpu().numpy() for x in neurotypical_label_array])
neurotypical_preds = np.array([x.detach().cpu().numpy() for x in neurotypical_pred_array])
f1_per_label_neurotypical = f1_score(neurotypical_labels, neurotypical_preds, average=None)

print(f"\nNeurotypical")
print(f"Video Test Neurotypical Accuracy: {accuracy_neurotypical:.6f}")
print(f"Video Test F1_Micro: {f1_micro_neurotypical:.5f}")
print(f"Video Test F1_Macro: {f1_macro_neurotypical:.5f}")
print(f"Video Test F1_Weighted: {f1_weighted_neurotypical:.5f}")
print(f"Video Test F1 Per Emotion for Neurotypical:")
for l, f in zip(label_emotion_names, f1_per_label_neurotypical):
    print(f"{l}: {f:.5f}")


# Test Neurodivergent Metrics Only
neurodivergent_pred_temp = torch.cat(neurodivergent_pred_array, dim=0)
neurodivergent_label_temp = torch.cat(neurodivergent_label_array, dim=0)
neurodivergent_preds_array = neurodivergent_pred_temp.view(-1).detach().cpu().numpy()
neurodivergent_labels_array = neurodivergent_label_temp.view(-1).detach().cpu().numpy()

accuracy_neurodivergent = accuracy_score(neurodivergent_labels_array, neurodivergent_preds_array)
f1_micro_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='micro')
f1_macro_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='macro')
f1_weighted_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='weighted')
neurodivergent_labels = np.array([x.detach().cpu().numpy() for x in neurodivergent_label_array])
neurodivergent_preds = np.array([x.detach().cpu().numpy() for x in neurodivergent_pred_array])
f1_per_label_neurodivergent = f1_score(neurodivergent_labels, neurodivergent_preds, average=None)

print(f"\nNeurodivergent")
print(f"Video Test Neurodivergent Accuracy: {accuracy_neurodivergent:.6f}")
print(f"Video Test F1_Micro: {f1_micro_neurodivergent:.5f}")
print(f"Video Test F1_Macro: {f1_macro_neurodivergent:.5f}")
print(f"Video Test F1_Weighted: {f1_weighted_neurodivergent:.5f}")
print(f"Video Test F1 Per Emotion for Neurodivergent:")
for l, f in zip(label_emotion_names, f1_per_label_neurodivergent):
    print(f"{l}: {f:.5f}")



"""
# Whole Dataset
weights_file = f"video_weights_epoch{num_epochs}_emotions_mlc.pth"
checkpoint = torch.load(weights_file)
model.load_state_dict(checkpoint['model_state_dict'])
head.load_state_dict(checkpoint['head_state_dict'])

whole_vid_data_list, whole_label_list, whole_condition_list = make_whole_dataset()
whole_dataset = VideoDataset(whole_vid_data_list, whole_label_list, whole_condition_list)
whole_loader = DataLoader(whole_dataset, batch_size=4)

model.eval()
video_embedding = []
whole_pred_array = []
whole_label_array = []
neurotypical_pred_array = []
neurotypical_label_array = []
neurodivergent_pred_array = []
neurodivergent_label_array = []
total_loss = 0.0
with torch.no_grad():
    for video_data, label, cond_label in Bar(whole_loader):
        video_data = video_data.to(device)
        label = label.to(device)

        outputs = model(video_data)
        cls = outputs.last_hidden_state[:, 0, :]
        pred = head(cls)
        loss = criterion(pred, label)
        
        prob = torch.sigmoid(pred)
        pred = (prob > 0.35).float()
        whole_pred_array.append(pred.detach())
        whole_label_array.append(label.detach())
        
        for i in range(cond_label.shape[0]):
            if cond_label[i] == 1:
                neurodivergent_pred_array.append(pred[i].detach())
                neurodivergent_label_array.append(label[i].detach())
            elif cond_label[i] == 0:
                neurotypical_pred_array.append(pred[i].detach())
                neurotypical_label_array.append(label[i].detach())

        video_embedding.append(cls.tolist())
        total_loss += loss.item()

whole_pred_temp = torch.cat(whole_pred_array, dim=0)
whole_label_temp = torch.cat(whole_label_array, dim=0)
whole_preds_array = whole_pred_temp.view(-1).detach().cpu().numpy()
whole_labels_array = whole_label_temp.view(-1).detach().cpu().numpy()

overall_accuracy = accuracy_score(whole_labels_array, whole_preds_array)
f1_micro = f1_score(whole_labels_array, whole_preds_array, average='micro')
f1_macro = f1_score(whole_labels_array, whole_preds_array, average='macro')
f1_weighted = f1_score(whole_labels_array, whole_preds_array, average='weighted')
f1_per_label = f1_score(whole_label_temp.squeeze(1).detach().cpu().numpy(), whole_pred_temp.squeeze(1).detach().cpu().numpy(), average=None)
overall_loss = total_loss / len(whole_loader)

label_emotion_names = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
label_attribute_names = ["Valence", "Arousal", "Dominance"]

print(f"Predictions: {pred}")
print(f"Labels: {label}")
print(f"Accuracy: {overall_accuracy:.6f}")
print(f"Loss: {overall_loss:.6f}")
print(f"F1_Micro: {f1_micro:.5f}")
print(f"F1_Macro: {f1_macro:.5f}")
print(f"F1_Weighted: {f1_weighted:.5f}")
print(f"F1 Per Emotion:")
for l, f in zip(label_emotion_names, f1_per_label):
    print(f"{l}: {f:.5f}")


# Evaluate Neurotypical
neurotypical_pred_temp = torch.cat(neurotypical_pred_array, dim=0)
neurotypical_label_temp = torch.cat(neurotypical_label_array, dim=0)
neurotypical_preds_array = neurotypical_pred_temp.view(-1).detach().cpu().numpy()
neurotypical_labels_array = neurotypical_label_temp.view(-1).detach().cpu().numpy()

overall_accuracy_neurotypical = accuracy_score(neurotypical_labels_array, neurotypical_preds_array)
f1_micro_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='micro')
f1_macro_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='macro')
f1_weighted_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='weighted')
neurotypical_labels = np.array([x.detach().cpu().numpy() for x in neurotypical_label_array])
neurotypical_preds = np.array([x.detach().cpu().numpy() for x in neurotypical_pred_array])
f1_per_label_neurotypical = f1_score(neurotypical_labels, neurotypical_preds, average=None)

print(f"\nNeurotypical")
print(f"Neurotypical Accuracy: {overall_accuracy_neurotypical:.6f}")
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

overall_accuracy_neurodivergent = accuracy_score(neurodivergent_labels_array, neurodivergent_preds_array)
f1_micro_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='micro')
f1_macro_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='macro')
f1_weighted_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='weighted')
neurodivergent_labels = np.array([x.detach().cpu().numpy() for x in neurodivergent_label_array])
neurodivergent_preds = np.array([x.detach().cpu().numpy() for x in neurodivergent_pred_array])
f1_per_label_neurodivergent = f1_score(neurodivergent_labels, neurodivergent_preds, average=None)

print(f"\nNeurodivergent")
print(f"Neurodivergent Accuracy: {overall_accuracy_neurodivergent:.6f}")
print(f"F1_Micro: {f1_micro_neurodivergent:.5f}")
print(f"F1_Macro: {f1_macro_neurodivergent:.5f}")
print(f"F1_Weighted: {f1_weighted_neurodivergent:.5f}")
print(f"F1 Per Emotion for Neurodivergent:")
for l, f in zip(label_emotion_names, f1_per_label_neurodivergent):
    print(f"{l}: {f:.5f}")


    
    
# Save Video Embeddings
video_embedding_file_npy = "C:/Users/User/PycharmProjects/Research Project/video_embeddings_pretrained_emotions_mlc.npy"
video_embedding = list(chain.from_iterable(video_embedding))
video_embedding = np.asarray(video_embedding, dtype=np.float32)
video_embedding = np.squeeze(video_embedding)
# print(video_embedding)
print(video_embedding.shape)
# np.save(video_embedding_file_npy, video_embedding)
"""







