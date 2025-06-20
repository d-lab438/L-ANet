import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import csv
import os
import time
import copy
import mne
import math
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay,recall_score, f1_score, precision_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

file_path = 'file_path'
all_file_names = os.listdir(file_path)
labels = []
features = []
filtered_sets = []
n_expend = 100
data = []

set_files = [file for file in all_file_names if file.endswith('.set')]
filtered_sets.extend(set_files)
print(filtered_sets)
for set_file in set_files:
    parts = set_file.split('-')
    if len(parts) > 2:
        labels.extend(parts[2][0] * n_expend)
print(labels)
labels = np.array(labels)
labels = labels.astype(int)
valid_labels = [1, 2, 3]
labels = labels[np.isin(labels, valid_labels)]
labels = labels - 1
onehot_encoder = OneHotEncoder(sparse=False)
labels_onehot = onehot_encoder.fit_transform(labels.reshape(-1, 1))

def load_eeg_data_seg(file_path, set_name):
    raw = mne.io.read_raw_eeglab(file_path + '/' + set_name, preload=True)
    total_duration = raw.times[-1]
    segment_duration = total_duration / n_expend
    for i in range(n_expend):
        start_time = i * segment_duration
        stop_time = (i + 1) * segment_duration
        segment = raw.copy().crop(tmin=start_time, tmax=stop_time)
        data.append(segment.get_data())
        
for set_name in filtered_sets:
    load_eeg_data_seg(file_path, set_name)
data = np.array(data)
assert len(data) == len(labels), f"Data and labels size mismatch: {len(data)} vs {len(labels)}"
n_samples, n_channels, n_timepoints = data.shape
data_reshaped = data.reshape(n_samples, -1)
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(data_reshaped)
X_scaled = X_scaled.reshape(n_samples, n_channels, n_timepoints)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
val_loss_history = []

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        return F.relu(x)


class SelfAttention1D(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention1D, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, length = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, length).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, length)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, length)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, length)
        out = self.gamma * out + x
        return out


class CNN1DModel(nn.Module):
    def __init__(self, num_classes):
        super(CNN1DModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=20, out_channels=64, kernel_size=7, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        self.resblock1 = ResidualBlock1D(in_channels=256,out_channels=256)

        self.attention = SelfAttention1D(256)
        self.flatten_dim = 3840
        
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = self.resblock1(x)
        x = self.attention(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 定义训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    model.train()
    best_accuracy = 0.0
    best_epoch = 0
    best_cm = None 
    best_model_path = 'best_model.pth'

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:

            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss_history.append(epoch_loss)

        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad()
            for val_inputs, val_labels in train_loader:
                val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()
                val_outputs = model(val_inputs)
                _, predicted = torch.max(val_outputs, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

                all_labels.extend(val_labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        train_accuracy = correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        cm = confusion_matrix(all_labels, all_preds)

        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            best_epoch = epoch
            best_cm = cm
            print(best_cm)
            torch.save(model.state_dict(), best_model_path)
            print(f'新最佳准确度: {best_accuracy:.4f}, 模型权重已保存')

        model.train()

    return epoch_loss,best_accuracy,best_cm

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = accuracy_score(all_labels, all_preds)

    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

    return val_loss, val_accuracy

batch_size = 64
X = X_scaled
y = labels_encoded
fold_results = []
cm_list = []
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold + 1}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = CNN1DModel(num_classes=3).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    num_epochs = 100

    train_loss,train_accuracy,best_cm=train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    val_loss,val_accuracy=validate_model(model, val_loader, criterion)

    fold_results.append({
        'fold': fold + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
    })
    cm_list.append(best_cm)

average_train_loss = np.mean([result['train_loss'] for result in fold_results])
average_val_loss = np.mean([result['val_loss'] for result in fold_results])
average_train_accuracy = np.mean([result['train_accuracy'] for result in fold_results])
average_val_accuracy = np.mean([result['val_accuracy'] for result in fold_results])

print(f"Average Train Loss: {average_train_loss:.4f}")
print(f"Average Val Loss: {average_val_loss:.4f}")
print(f"Average Train Accuracy: {average_train_accuracy:.4f}")
print(f"Average Val Accuracy: {average_val_accuracy:.4f}")
