import numpy as np
import pandas as pd
import os
import librosa
import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from barbar import Bar
import sys
import csv
from itertools import chain
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from natsort import natsorted
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix
np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    # Convert to Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    # Resample to 16 kHz
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    return waveform


def get_video_number(file_path):
    file = os.path.basename(file_path)
    video_number = file.split("eo")[1]
    video_number = int(video_number.split(".")[0])
    return video_number


def get_corresponding_data(video_number):
    audio_folder = "C:/Users/User/OneDrive/Documents/Research Project Audio Files/"
    audio_filename = "Video" + str(video_number) + ".mp3"
    audio_file = os.path.join(audio_folder, audio_filename)

    labels_file = "C:/Users/User/PycharmProjects/Research Project/New_Labels_By_Classification_Emotions_Threshold15.npy"
    # labels_file = "C:/Users/User/PycharmProjects/Research Project/Revised_New_Labels_By_Classification_Attributes.npy"
    labels_data = np.load(labels_file)
    label_clip = labels_data[video_number - 1]
    label_clip = label_clip.astype(float)

    condition_label_file = "C:/Users/User/PycharmProjects/Research Project/Condition_Labels.csv"
    condition_label = pd.read_csv(condition_label_file)
    cond_label = condition_label['Neurodivergent'].loc[video_number - 1]
    # Yes (1): Autism/Neurodivergent, No (0): Normal/Neurotypical

    return audio_file, label_clip, cond_label


def make_whole_dataset():
    whole_audio_file_list = []
    whole_label_list = []
    whole_condition_list = []
    for i in range(1000):
        audio_file, label_clip, cond_label = get_corresponding_data(i + 1)
        whole_audio_file_list.append(audio_file)
        whole_label_list.append(label_clip)
        whole_condition_list.append(cond_label)
    return whole_audio_file_list, whole_label_list, whole_condition_list


class AudioDataset(Dataset):
    def __init__(self, audio_filepaths, labels, cond_labels):
        self.audio_filepaths = audio_filepaths
        self.labels = labels
        self.cond_labels = cond_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        waveform = load_audio(self.audio_filepaths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        cond_label = self.cond_labels[idx]
        return {
            "input_values": waveform,
            "labels": label,
            "cond_labels": cond_label
        }


def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    cond_labels = [item["cond_labels"] for item in batch]

    input_values = [x.squeeze().numpy() for x in input_values]

    inputs = pretrained_processor(input_values, sampling_rate=16000, return_tensors="pt", padding=True, truncation=True, max_length=160000)
    labels = torch.stack(labels)
    cond_labels = torch.tensor(cond_labels)

    return inputs, labels, cond_labels


def get_split_data(phase_split, phase_file_list):
    phase_split_audio_file_list = []
    phase_split_label_list = []
    phase_split_condition_list = []
    for file in phase_file_list:
        video_number = get_video_number(file)
        audio_file, new_label_clip, cond_label = get_corresponding_data(video_number)
        phase_split_audio_file_list.append(audio_file)
        phase_split_label_list.append(new_label_clip)
        phase_split_condition_list.append(cond_label)
    return phase_split_audio_file_list, phase_split_label_list, phase_split_condition_list


def return_train_and_valid_data(train_validation_split, epoch):
    remainder = epoch % 5
    split_size = int(len(train_validation_split) / 5)
    start_index = int(remainder * (split_size))
    final_index = int((remainder + 1) * (split_size))
    val_dataset = train_validation_split[start_index:final_index]

    train_dataset = []
    for file in train_validation_split:
        if file not in val_dataset:
            train_dataset.append(file)

    train_split_audio_file_list, train_split_label_list, train_split_condition_list = get_split_data("Train", train_dataset)
    validation_split_audio_file_list, validation_split_label_list, validation_split_condition_list = get_split_data("Validation", val_dataset)

    return train_split_audio_file_list, train_split_label_list, train_split_condition_list, validation_split_audio_file_list, validation_split_label_list, validation_split_condition_list


# Load Pretrained Wav2Vec2
pretrained_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("superb/wav2vec2-base-superb-er")

train_validation_split_file = "C:/Users/User/PycharmProjects/Research Project/Train_Validation_Split.csv"
train_validation_split = np.loadtxt(train_validation_split_file, delimiter=',', dtype=str)
train_validation_split = train_validation_split.tolist()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
classifier = nn.Linear(768, 7).to(device)
model.to(device)


for epoch in range(num_epochs):
    model.train()
    print(f"Epoch: {epoch+1}")

    # Get Train and Validation Split Data
    train_split_audio_file_list, train_split_label_list, train_split_condition_list, validation_split_audio_file_list, validation_split_label_list, validation_split_condition_list = return_train_and_valid_data(train_validation_split, epoch)

    # Create Train and Validation Datasets
    train_dataset = AudioDataset(train_split_audio_file_list, train_split_label_list, train_split_condition_list)
    val_dataset = AudioDataset(validation_split_audio_file_list, validation_split_label_list, validation_split_condition_list)

    # Create Train and Validation Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)

    # Training
    total_train_loss = 0.0
    train_pred_array = []
    train_label_array = []
    for audio_input, label, cond_label in Bar(train_loader):
        inputs = {k: v.to(device).float() for k, v in audio_input.items()}
        label = label.to(device)

        output = model(**inputs)
        hidden_states = output.last_hidden_state
        embeddings = hidden_states.mean(dim=1)
        logits = classifier(embeddings)

        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        prob = torch.sigmoid(logits)
        pred = (prob > 0.35).float()

        train_pred_array.append(pred.detach())
        train_label_array.append(label.detach())

        total_train_loss += loss.item()

    train_pred_temp = torch.cat(train_pred_array, dim=0)
    train_label_temp = torch.cat(train_label_array, dim=0)
    train_preds_array = train_pred_temp.view(-1).cpu().numpy()
    train_labels_array = train_label_temp.view(-1).cpu().numpy()

    train_accuracy = accuracy_score(train_labels_array, train_preds_array)
    train_loss = total_train_loss / len(train_loader)
    print(f"Audio Epoch {epoch+1}, Train Accuracy: {train_accuracy:.5f}, Train Loss: {train_loss:.6f}")

    # Validation
    model.eval()
    total_val_loss = 0.0
    val_pred_array = []
    val_label_array = []
    with torch.no_grad():
        for audio_input, label, cond_label in Bar(val_loader):
            inputs = {k: v.to(device) for k, v in audio_input.items()}
            label = label.to(device)

            output = model(**inputs)
            hidden_states = output.last_hidden_state
            embeddings = hidden_states.mean(dim=1)
            logits = classifier(embeddings)
            loss = criterion(logits, label)

            prob = torch.sigmoid(logits)
            pred = (prob > 0.35).float()
            val_pred_array.append(pred.detach())
            val_label_array.append(label.detach())

            total_val_loss += loss.item()

        val_pred_temp = torch.cat(val_pred_array, dim=0)
        val_label_temp = torch.cat(val_label_array, dim=0)
        val_preds_array = val_pred_temp.view(-1).cpu().numpy()
        val_labels_array = val_label_temp.view(-1).cpu().numpy()

        val_accuracy = accuracy_score(val_labels_array, val_preds_array)
        val_loss = total_val_loss / len(val_loader)
        print(f"Audio Epoch {epoch+1}, Val Accuracy: {val_accuracy:.5f}, Val Loss: {val_loss:.6f}")

    # Save latest model
    if epoch == (num_epochs - 1):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'loss': val_loss
        }, f"audio_weights_epoch{num_epochs}_attributes_mlc_wav2vec2.pth")


# Evaluation
weights_file = f"audio_weights_epoch{num_epochs}_attributes_mlc_wav2vec2.pth"
checkpoint = torch.load(weights_file)
model.load_state_dict(checkpoint['model_state_dict'])
classifier.load_state_dict(checkpoint['classifier_state_dict'])

# Get Test Split Data
test_split_file = "C:/Users/User/PycharmProjects/Research Project/Test_Split.csv"
test_split = np.loadtxt(test_split_file, delimiter=',', dtype=str)
test_split = test_split.tolist()
test_split_audio_file_list, test_split_label_list, test_split_condition_list = get_split_data("Test", test_split)

# Create Test Audio Dataset
test_dataset = AudioDataset(test_split_audio_file_list, test_split_label_list, test_split_condition_list)

# Load Test Dataloader
test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)

# Testing
total_test_loss = 0.0
model.eval()
test_pred_array = []
test_label_array = []
neurotypical_pred_array = []
neurotypical_label_array = []
neurodivergent_pred_array = []
neurodivergent_label_array = []
with torch.no_grad():
    for audio_input, label, cond_label in Bar(test_loader):
        inputs, label = audio_input, label
        inputs = {k: v.to(device) for k, v in inputs.items()}
        label = label.to(device)

        output = model(**inputs)
        hidden_states = output.last_hidden_state
        embeddings = hidden_states.mean(dim=1)
        logits = classifier(embeddings)
        loss = criterion(logits, label)

        prob = torch.sigmoid(logits)
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
test_f1_per_label = f1_score(test_label_temp.squeeze(1).cpu().numpy(), test_pred_temp.squeeze(1).cpu().numpy(), average=None)
test_loss = total_test_loss / len(test_loader)

label_emotion_names = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
label_attribute_names = ["Valence", "Arousal", "Dominance"]

print(f"Audio Test Predictions: {pred}")
print(f"Audio Test Labels: {label}")
print(f"Audio Test Accuracy: {test_accuracy:.5f}")
print(f"Audio Test Loss: {test_loss:.6f}")
print(f"Audio Test F1_Micro: {test_f1_micro:.5f}")
print(f"Audio Test F1_Macro: {test_f1_macro:.5f}")
print(f"Audio Test F1_Weighted: {test_f1_weighted:.5f}")
print(f"Audio Test F1 Per Emotion:")
for l, f in zip(label_emotion_names, test_f1_per_label):
    print(f"{l}: {f:.5f}")


# Test Neurotypical Metrics Only
neurotypical_pred_temp = torch.cat(neurotypical_pred_array, dim=0)
neurotypical_label_temp = torch.cat(neurotypical_label_array, dim=0)
neurotypical_preds_array = neurotypical_pred_temp.view(-1).cpu().numpy()
neurotypical_labels_array = neurotypical_label_temp.view(-1).cpu().numpy()

label_emotion_names = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
label_attribute_names = ["Valence", "Arousal", "Dominance"]

overall_accuracy_neurotypical = accuracy_score(neurotypical_labels_array, neurotypical_preds_array)
f1_micro_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='micro')
f1_macro_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='macro')
f1_weighted_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='weighted')
neurotypical_labels = np.array([x.detach().cpu().numpy() for x in neurotypical_label_array])
neurotypical_preds = np.array([x.detach().cpu().numpy() for x in neurotypical_pred_array])
f1_per_label_neurotypical = f1_score(neurotypical_labels, neurotypical_preds, average=None)

print(f"\nNeurotypical")
print(f"Accuracy: {overall_accuracy_neurotypical:.6f}")
print(f"F1_Micro: {f1_micro_neurotypical:.5f}")
print(f"F1_Macro: {f1_macro_neurotypical:.5f}")
print(f"F1_Weighted: {f1_weighted_neurotypical:.5f}")
print(f"F1 Per Emotion for Neurotypical:")
for l, f in zip(label_emotion_names, f1_per_label_neurotypical):
    print(f"{l}: {f:.5f}")


# Test Neurodivergent Metrics Only
neurodivergent_pred_temp = torch.cat(neurodivergent_pred_array, dim=0)
neurodivergent_label_temp = torch.cat(neurodivergent_label_array, dim=0)
neurodivergent_preds_array = neurodivergent_pred_temp.view(-1).cpu().numpy()
neurodivergent_labels_array = neurodivergent_label_temp.view(-1).cpu().numpy()

overall_accuracy_neurodivergent = accuracy_score(neurodivergent_labels_array, neurodivergent_preds_array)
f1_micro_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='micro')
f1_macro_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='macro')
f1_weighted_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='weighted')
neurodivergent_labels = np.array([x.detach().cpu().numpy() for x in neurodivergent_label_array])
neurodivergent_preds = np.array([x.detach().cpu().numpy() for x in neurodivergent_pred_array])
f1_per_label_neurodivergent = f1_score(neurodivergent_labels, neurodivergent_preds, average=None)

print(f"\nNeurodivergent")
print(f"Accuracy: {overall_accuracy_neurodivergent:.6f}")
print(f"F1_Micro: {f1_micro_neurodivergent:.5f}")
print(f"F1_Macro: {f1_macro_neurodivergent:.5f}")
print(f"F1_Weighted: {f1_weighted_neurodivergent:.5f}")
print(f"F1 Per Emotion for Neurodivergent:")
for l, f in zip(label_emotion_names, f1_per_label_neurodivergent):
    print(f"{l}: {f:.5f}")



"""
# Whole File
weights_file = f"audio_weights_epoch{num_epochs}_attributes_mlc_wav2vec2.pth"
checkpoint = torch.load(weights_file)
model.load_state_dict(checkpoint['model_state_dict'])
classifier.load_state_dict(checkpoint['classifier_state_dict'])

# Get Whole Dataset
whole_audio_file_list, whole_label_list, whole_condition_list = make_whole_dataset()
whole_dataset = AudioDataset(whole_audio_file_list, whole_label_list, whole_condition_list)

# Load Whole DataLoader
whole_loader = DataLoader(whole_dataset, batch_size=4, collate_fn=collate_fn)

audio_embedding = []
whole_pred_array = []
whole_label_array = []
neurotypical_pred_array = []
neurotypical_label_array = []
neurodivergent_pred_array = []
neurodivergent_label_array = []
total_loss = 0.0
model.eval()
model.to(device)
classifier.to(device)
# Testing
with torch.no_grad():
    for audio_input, label, cond_label in Bar(whole_loader):
        inputs = {k: v.to(device) for k, v in audio_input.items()}
        label = label.to(device)

        output = model(**inputs)
        hidden_states = output.last_hidden_state
        embeddings = hidden_states.mean(dim=1)
        logits = classifier(embeddings)
        loss = criterion(logits, label)

        prob = torch.sigmoid(logits)
        pred = (prob > 0.35).float()

        audio_embedding.append(embeddings.squeeze().tolist())
        whole_pred_array.append(pred.detach())
        whole_label_array.append(label.detach())
        
        for i in range(cond_label.shape[0]):
            if cond_label[i] == 1:
                neurodivergent_pred_array.append(pred[i].detach())
                neurodivergent_label_array.append(label[i].detach())
            elif cond_label[i] == 0:
                neurotypical_pred_array.append(pred[i].detach())
                neurotypical_label_array.append(label[i].detach())

        total_loss += loss.item()

whole_pred_temp = torch.cat(whole_pred_array, dim=0)
whole_label_temp = torch.cat(whole_label_array, dim=0)
whole_preds_array = whole_pred_temp.view(-1).cpu().numpy()
whole_labels_array = whole_label_temp.view(-1).cpu().numpy()

overall_accuracy = accuracy_score(whole_labels_array, whole_preds_array)
f1_micro = f1_score(whole_labels_array, whole_preds_array, average='micro')
f1_macro = f1_score(whole_labels_array, whole_preds_array, average='macro')
f1_weighted = f1_score(whole_labels_array, whole_preds_array, average='weighted')
f1_per_label = f1_score(whole_label_temp.squeeze(1).cpu().numpy(), whole_pred_temp.squeeze(1).cpu().numpy(), average=None)
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
neurotypical_preds_array = neurotypical_pred_temp.view(-1).cpu().numpy()
neurotypical_labels_array = neurotypical_label_temp.view(-1).cpu().numpy()

overall_accuracy_neurotypical = accuracy_score(neurotypical_labels_array, neurotypical_preds_array)
f1_micro_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='micro')
f1_macro_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='macro')
f1_weighted_neurotypical = f1_score(neurotypical_labels_array, neurotypical_preds_array, average='weighted')
neurotypical_labels = np.array([x.detach().cpu().numpy() for x in neurotypical_label_array])
neurotypical_preds = np.array([x.detach().cpu().numpy() for x in neurotypical_pred_array])
f1_per_label_neurotypical = f1_score(neurotypical_labels, neurotypical_preds, average=None)

print(f"\nNeurotypical")
print(f"Accuracy: {overall_accuracy_neurotypical:.6f}")
print(f"F1_Micro: {f1_micro_neurotypical:.5f}")
print(f"F1_Macro: {f1_macro_neurotypical:.5f}")
print(f"F1_Weighted: {f1_weighted_neurotypical:.5f}")
print(f"F1 Per Emotion for Neurotypical:")
for l, f in zip(label_emotion_names, f1_per_label_neurotypical):
    print(f"{l}: {f:.5f}")


# Evaluate Neurodivergent
neurodivergent_pred_temp = torch.cat(neurodivergent_pred_array, dim=0)
neurodivergent_label_temp = torch.cat(neurodivergent_label_array, dim=0)
neurodivergent_preds_array = neurodivergent_pred_temp.view(-1).cpu().numpy()
neurodivergent_labels_array = neurodivergent_label_temp.view(-1).cpu().numpy()

overall_accuracy_neurodivergent = accuracy_score(neurodivergent_labels_array, neurodivergent_preds_array)
f1_micro_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='micro')
f1_macro_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='macro')
f1_weighted_neurodivergent = f1_score(neurodivergent_labels_array, neurodivergent_preds_array, average='weighted')
neurodivergent_labels = np.array([x.detach().cpu().numpy() for x in neurodivergent_label_array])
neurodivergent_preds = np.array([x.detach().cpu().numpy() for x in neurodivergent_pred_array])
f1_per_label_neurodivergent = f1_score(neurodivergent_labels, neurodivergent_preds, average=None)

print(f"\nNeurodivergent")
print(f"Accuracy: {overall_accuracy_neurodivergent:.6f}")
print(f"F1_Micro: {f1_micro_neurodivergent:.5f}")
print(f"F1_Macro: {f1_macro_neurodivergent:.5f}")
print(f"F1_Weighted: {f1_weighted_neurodivergent:.5f}")
print(f"F1 Per Emotion for Neurodivergent:")
for l, f in zip(label_emotion_names, f1_per_label_neurodivergent):
    print(f"{l}: {f:.5f}")


# Save Audio Embedding
audio_embedding_file_npy = "C:/Users/User/PycharmProjects/Research Project/audio_embeddings_pretrained_attributes_mlc_wav2vec2.npy"
audio_embedding = list(chain.from_iterable(audio_embedding))
audio_embedding = np.asarray(audio_embedding, dtype=np.float32)
audio_embedding = np.squeeze(audio_embedding)
# print(audio_embedding)
print(audio_embedding.shape)
# np.save(audio_embedding_file_npy, audio_embedding)
"""











