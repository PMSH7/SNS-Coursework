import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import os

M, N = 20, 4
R = 20
V = 0
MOB = 20
nb_MOB = 2

nb_data = 370 - (M + N)
InputSize = M + R + V + MOB*nb_MOB
OutputSize = N
HidenSize = 90
LR = 0.005

data_case_max = 81566
data_case_min = 22

new_cases = np.loadtxt('new_cases.csv', delimiter=",")
basic_reproduction_number = np.loadtxt('basic_reproduction_number.csv', delimiter=",")
people_vaccinated = np.loadtxt('people_vaccinated.csv', delimiter=",")
retail_and_recreation_percent_change_from_baseline = np.loadtxt('retail_and_recreation_percent_change_from_baseline.csv', delimiter=",")
grocery_and_pharmacy_percent_change_from_baseline = np.loadtxt('grocery_and_pharmacy_percent_change_from_baseline.csv', delimiter=",")
parks_percent_change_from_baseline = np.loadtxt('parks_percent_change_from_baseline.csv', delimiter=",")
transit_stations_percent_change_from_baseline = np.loadtxt('transit_stations_percent_change_from_baseline.csv', delimiter=",")
workplaces_percent_change_from_baseline = np.loadtxt('workplaces_percent_change_from_baseline.csv', delimiter=",")
residential_percent_change_from_baseline = np.loadtxt('residential_percent_change_from_baseline.csv', delimiter=",")

data = np.zeros((nb_data, InputSize + OutputSize))
for row in range(nb_data):
    data[row,:R] = basic_reproduction_number[row]
    data[row,R+V+MOB*0:R+V+MOB*1] = grocery_and_pharmacy_percent_change_from_baseline[row]
    data[row,R+V+MOB*1:R+V+MOB*2] = transit_stations_percent_change_from_baseline[row]
    data[row,R+V+MOB*nb_MOB:] = new_cases[row]

with open('covid_data.csv', 'wb') as f:
    np.savetxt(f, data, delimiter=",")

train_ratio = 0.8
val_ratio = 0.1

train_split_point = int(nb_data * train_ratio)
val_split_point = int(nb_data * (train_ratio + val_ratio))
train_data = data[:train_split_point, :InputSize]
train_label = data[:train_split_point, InputSize:]
val_data = data[train_split_point:val_split_point, :InputSize]
val_label = data[train_split_point:val_split_point, InputSize:]

class CovidDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.0)
        self.fc2 = nn.Linear(hidden_size, output_size)  
    
    def forward(self, x):
        x = x.to(torch.float32)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

ITER = 200
batch_size = 32

train_dataset = CovidDataset(train_data, train_label)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = CovidDataset(val_data, val_label)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
model = NeuralNet(input_size=InputSize, hidden_size=HidenSize, output_size=OutputSize)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LR)

loss_log_iter = []
mae_log_iter = []
min_val = 1.0E20
for iter in range(ITER):
    model.train()
    loss_log = []
    mae_log = []
    for idx, (train, label) in enumerate(train_dataloader):
        predict = model(train)
        label = label.to(torch.float32)
        loss = loss_func(predict, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_log.append(loss.item())
        mae_log.append(torch.mean(torch.abs(predict - label) / (label)).item())
    loss_mean = np.mean(np.array(loss_log))
    loss_log_iter.append(loss_mean)
    mae_mean = np.mean(np.array(mae_log))
    mae_log_iter.append(mae_mean)

    model.eval()
    with torch.no_grad():
        predict_value = []
        label_value = []
        for id, (validate, label) in enumerate(val_dataloader):
            predict = model(validate)
            predict_value.append(predict.squeeze().detach().numpy())
            label_value.append(label.squeeze().detach().numpy())

        predict_value = np.array(predict_value)
        re_predict_value = predict_value * (data_case_max - data_case_min) + data_case_min
        label_value = np.array(label_value)
        re_label_value = label_value * (data_case_max - data_case_min) + data_case_min

        val = np.mean(abs(re_predict_value - re_label_value) / (re_label_value)) 
        if val < min_val:
            min_val = val
            torch.save(model.state_dict(), "param.pth")
    print(f'Iter num: {iter}, loss: {loss_mean}, val: {val}')