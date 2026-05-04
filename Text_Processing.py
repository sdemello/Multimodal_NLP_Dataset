import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import os
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import csv
import ast
import h5py
from barbar import Bar
from natsort import natsorted
from itertools import chain
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix


def get_video_number(file_path):
    file = os.path.basename(file_path)
    video_number = file.split("eo")[1]
    video_number = int(video_number.split(".")[0])
    return video_number


def normalize_label(label):
    new_label = label / 100.0
    # new_label = (new_label + 3) / 6
    return new_label


def get_corresponding_data(video_number, input_ids, attention_masks):
    input_id = input_ids[video_number]
    attention_mask = attention_masks[video_number]

    labels_file = "C:/Users/User/PycharmProjects/Research Project/New_Labels_By_Classification_Emotions_Threshold15.npy"
    labels_data = np.load(labels_file)
    label_clip = labels_data[video_number]
    label_clip = label_clip.astype(float)

    condition_label_file = "C:/Users/User/PycharmProjects/Research Project/Condition_Labels.csv"
    condition_label = pd.read_csv(condition_label_file)
    cond_label = condition_label['Neurodivergent'].loc[video_number]
    # Yes (1): Autism/Neurodivergent, No (0): Normal/Neurotypical

    return input_id, attention_mask, label_clip, cond_label


def make_whole_dataset(input_ids, attention_masks):
    whole_text_input_ids_list = []
    whole_attention_masks_list = []
    whole_label_list = []
    whole_condition_list = []
    for i in range(1000):
        text_input_id, attention_mask, new_label_clip, condition_label = get_corresponding_data(i, input_ids, attention_masks)
        whole_text_input_ids_list.append(text_input_id)
        whole_attention_masks_list.append(attention_mask)
        whole_label_list.append(new_label_clip)
        whole_condition_list.append(condition_label)
    return whole_text_input_ids_list, whole_attention_masks_list, whole_label_list, whole_condition_list


def get_split_data(phase_split, phase_file_list, input_ids, attention_masks):
    phase_split_text_input_ids_list = []
    phase_split_attention_masks_list = []
    phase_split_label_list = []
    phase_split_condition_list = []
    for file in phase_file_list:
        video_number = get_video_number(file)
        text_input_id, attention_mask, new_label_clip, condition_label = get_corresponding_data(video_number-1, input_ids, attention_masks)
        phase_split_text_input_ids_list.append(text_input_id)
        phase_split_attention_masks_list.append(attention_mask)
        phase_split_label_list.append(new_label_clip)
        phase_split_condition_list.append(condition_label)
    return phase_split_text_input_ids_list, phase_split_attention_masks_list, phase_split_label_list, phase_split_condition_list


def return_train_and_valid_indices(train_validation_split, epoch):
    remainder = epoch % 5
    split_size = int(len(train_validation_split) / 5)
    indices = np.arange(len(train_validation_split))
    valid_indices = indices[(remainder*(split_size)):((remainder+1)*split_size)]
    train_indices = np.concatenate([indices[:(remainder*split_size)], indices[((remainder+1)*split_size):]])
    return train_indices, valid_indices


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = base_model
        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 7)
        )

    def forward(self, inputs_ids, attention_mask, return_embedding=False):
        outputs = self.bert(input_ids=inputs_ids, attention_mask=attention_mask)
        x_embedding = outputs.last_hidden_state.mean(dim=1)
        x = self.fc(x_embedding)
        if return_embedding is True:
            return x, x_embedding
        return x


class TextDataset(Dataset):
    def __init__(self, text_input_ids, attention_masks, labels, cond_labels):
        super().__init__()
        self.text_input_ids = text_input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.cond_labels = cond_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text_input = self.text_input_ids[idx]
        attention_mask = self.attention_masks[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        cond_label = self.cond_labels[idx]
        return text_input, attention_mask, label, cond_label


text_preprocessing_file = "C:/Users/User/Downloads/Text_Preprocessing.txt"
with open(text_preprocessing_file, 'r') as f:
    content = f.read()
    text_preprocessing_data = ast.literal_eval(content)
text_inputs = tokenizer(text_preprocessing_data, padding=True, truncation=True, return_tensors="pt")
text_input_ids = text_inputs['input_ids']
attention_masks = text_inputs['attention_mask']

train_validation_file = "C:/Users/User/Downloads/Train_Validation_Split.csv"
train_validation_split = np.loadtxt(train_validation_file, delimiter=',', dtype=str)
train_validation_split = train_validation_split.tolist()
train_validation_split_text_input_ids_list, train_validation_split_attention_masks_list, train_validation_split_label_list, train_validation_split_condition_list = get_split_data("Train-Validation", train_validation_split, text_input_ids, attention_masks)
train_validation_dataset = TextDataset(train_validation_split_text_input_ids_list, train_validation_split_attention_masks_list, train_validation_split_label_list, train_validation_split_condition_list)

model = TextModel()
# criterion = nn.SmoothL1Loss()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 20

"""
model.train()
for epoch in range(num_epochs):
    model.train()
    print(f"Epoch {epoch+1}")

    # Get Train and Validation Datasets
    train_indices, valid_indices = return_train_and_valid_indices(train_validation_split, epoch)
    train_dataset = Subset(train_validation_dataset, train_indices)
    valid_dataset = Subset(train_validation_dataset, valid_indices)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16)

    train_total_loss = 0.0
    train_pred_array = []
    train_label_array = []
    for text_input_id, attention_mask, label, cond_label in Bar(train_loader):
        output = model(text_input_id, attention_mask, return_embedding=False)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prob = torch.sigmoid(output)
        pred = (prob > 0.35).float()
        train_pred_array.append(pred.detach())
        train_label_array.append(label.detach())

        train_total_loss += loss.item()

    train_pred_temp = torch.cat(train_pred_array, dim=0)
    train_label_temp = torch.cat(train_label_array, dim=0)
    train_preds_array = train_pred_temp.view(-1).numpy()
    train_labels_array = train_label_temp.view(-1).numpy()

    train_accuracy = accuracy_score(train_labels_array, train_preds_array)
    train_loss = train_total_loss / len(train_loader)
    print(f"Text Epoch {epoch+1}, Train Accuracy: {train_accuracy:.5f}, Train Loss: {train_loss:.6f}")

    # Validation
    model.eval()
    valid_total_loss = 0.0
    val_pred_array = []
    val_label_array = []
    with torch.no_grad():
        for text_input_id, attention_mask, label, cond_label in Bar(valid_loader):
            output = model(text_input_id, attention_mask, return_embedding=False)
            loss = criterion(output, label)

            prob = torch.sigmoid(output)
            pred = (prob > 0.35).float()
            val_pred_array.append(pred.detach())
            val_label_array.append(label.detach())

            valid_total_loss += loss.item()

        val_pred_temp = torch.cat(val_pred_array, dim=0)
        val_label_temp = torch.cat(val_label_array, dim=0)
        val_preds_array = val_pred_temp.view(-1).numpy()
        val_labels_array = val_label_temp.view(-1).numpy()

        val_accuracy = accuracy_score(val_labels_array, val_preds_array)
        valid_loss = valid_total_loss / len(valid_loader)
        print(f"Text Epoch {epoch+1}, Val Accuracy: {val_accuracy:.5f}, Val Loss: {valid_loss:.6f}")

    if epoch == (num_epochs - 1):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }, f"text_weights_epoch{num_epochs}_emotions_mlc.pth")
"""


# weights_file = f"text_weights_epoch{num_epochs}_emotions_mlc.pth"
weights_file = f"C:/Users/User/Downloads/text_weights_epoch20_emotions_mlc.pth"
checkpoint = torch.load(weights_file)
model.load_state_dict(checkpoint['model_state_dict'])

# Get Test Split
test_split_file = "C:/Users/User/Downloads/Test_Split.csv"
test_split = np.loadtxt(test_split_file, delimiter=',', dtype=str)
test_split = test_split.tolist()
test_split_text_input_ids_list, test_split_attention_masks_list, test_split_label_list, test_split_condition_list = get_split_data("Test", test_split, text_input_ids, attention_masks)
test_dataset = TextDataset(test_split_text_input_ids_list, test_split_attention_masks_list, test_split_label_list, test_split_condition_list)
test_loader = DataLoader(test_dataset, batch_size=16)

# Testing
model.eval()
test_total_loss = 0.0
test_pred_array = []
test_label_array = []
neurotypical_pred_array = []
neurotypical_label_array = []
neurodivergent_pred_array = []
neurodivergent_label_array = []
with torch.no_grad():
    for text_input_id, attention_mask, label, cond_label in Bar(test_loader):
        output = model(text_input_id, attention_mask, return_embedding=False)
        loss = criterion(output, label)

        prob = torch.sigmoid(output)
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

        test_total_loss += loss.item()

test_pred_temp = torch.cat(test_pred_array, dim=0)
test_label_temp = torch.cat(test_label_array, dim=0)
test_preds_array = test_pred_temp.view(-1).detach().cpu().numpy()
test_labels_array = test_label_temp.view(-1).detach().cpu().numpy()

test_accuracy = accuracy_score(test_labels_array, test_preds_array)
test_f1_micro = f1_score(test_labels_array, test_preds_array, average='micro')
test_f1_macro = f1_score(test_labels_array, test_preds_array, average='macro')
test_f1_weighted = f1_score(test_labels_array, test_preds_array, average='weighted')
test_f1_per_label = f1_score(test_label_temp.squeeze(1).detach().cpu().numpy(), test_pred_temp.squeeze(1).detach().cpu().numpy(), average=None)
test_loss = test_total_loss / len(test_loader)

label_emotion_names = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
label_attribute_names = ["Valence", "Arousal", "Dominance"]

print(f"Text Test Predictions: {pred}")
print(f"Text Test Labels: {label}")
print(f"Text Test Accuracy: {test_accuracy:.5f}")
print(f"Text Test Loss: {test_loss:.6f}")
print(f"Text Test F1_Micro: {test_f1_micro:.5f}")
print(f"Text Test F1_Macro: {test_f1_macro:.5f}")
print(f"Text Test F1_Weighted: {test_f1_weighted:.5f}")
print(f"Text Test F1 Per Emotion:")
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
print(np.array(neurotypical_label_array).shape)
print(np.array(neurotypical_pred_array).shape)
neurotypical_labels = np.array([x.detach().cpu().numpy() for x in neurotypical_label_array])
neurotypical_preds = np.array([x.detach().cpu().numpy() for x in neurotypical_pred_array])
f1_per_label_neurotypical = f1_score(neurotypical_labels, neurotypical_preds, average=None)

print(f"\nNeurotypical")
print(f"Accuracy: {accuracy_neurotypical:.6f}")
print(f"F1_Micro: {f1_micro_neurotypical:.5f}")
print(f"F1_Macro: {f1_macro_neurotypical:.5f}")
print(f"F1_Weighted: {f1_weighted_neurotypical:.5f}")
print(f"F1 Per Emotion for Neurotypical:")
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
print(f"Accuracy: {accuracy_neurodivergent:.6f}")
print(f"F1_Micro: {f1_micro_neurodivergent:.5f}")
print(f"F1_Macro: {f1_macro_neurodivergent:.5f}")
print(f"F1_Weighted: {f1_weighted_neurodivergent:.5f}")
print(f"F1 Per Emotion for Neurodivergent:")
for l, f in zip(label_emotion_names, f1_per_label_neurodivergent):
    print(f"{l}: {f:.5f}")


"""
# Whole File
weights_file = f"text_weights_epoch{num_epochs}_emotions_mlc.pth"
checkpoint = torch.load(weights_file)
model.load_state_dict(checkpoint['model_state_dict'])

whole_text_input_ids_list, whole_attention_masks_list, whole_label_list, whole_condition_list = make_whole_dataset(text_input_ids, attention_masks)
whole_dataset = TextDataset(whole_text_input_ids_list, whole_attention_masks_list, whole_label_list, whole_condition_list)
whole_loader = DataLoader(whole_dataset, batch_size=16)

text_embedding = []
whole_pred_array = []
whole_label_array = []
neurotypical_pred_array = []
neurotypical_label_array = []
neurodivergent_pred_array = []
neurodivergent_label_array = []
total_loss = 0.0
with torch.no_grad():
    for text_input_id, attention_mask, label, cond_label in Bar(whole_loader):
        output, temp_text_embedding = model(text_input_id, attention_mask, return_embedding=True)
        loss = criterion(output, label)
        prob = torch.sigmoid(output)
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
        
        text_embedding.append(temp_text_embedding.tolist())
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
print(f"Accuracy: {overall_accuracy_neurodivergent:.6f}")
print(f"F1_Micro: {f1_micro_neurodivergent:.5f}")
print(f"F1_Macro: {f1_macro_neurodivergent:.5f}")
print(f"F1_Weighted: {f1_weighted_neurodivergent:.5f}")
print(f"F1 Per Emotion for Neurodivergent:")
for l, f in zip(label_emotion_names, f1_per_label_neurodivergent):
    print(f"{l}: {f:.5f}")
    

# Save Text Embeddings
text_embedding_file_npy = "/home/user2/text_embeddings_pretrained_emotions_mlc.npy"
text_embedding = list(chain.from_iterable(text_embedding))
text_embedding = np.asarray(text_embedding, dtype=np.float32)
text_embedding = np.squeeze(text_embedding)
# print(text_embedding)
print(text_embedding.shape)
# np.save(text_embedding_file_npy, text_embedding)
"""















