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

data = np.loadtxt('covid_data.csv', delimiter=",")

train_ratio = 0.8
val_ratio = 0.1
val_split_point = int(nb_data * (train_ratio + val_ratio))

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

model = NeuralNet(input_size=InputSize, hidden_size=HidenSize, output_size=OutputSize)
model.load_state_dict(torch.load("param.pth"))

whole_predict = []
whole_label = []
for test_cycle in range(0,19): 

    test_data = data[val_split_point+test_cycle:val_split_point+test_cycle+1, :InputSize]
    test_label = data[val_split_point+test_cycle:val_split_point+test_cycle+1, InputSize:]

    days_predict = []
    days_label = []
    for day in range(0,4):
        test_dataset = CovidDataset(test_data, test_label)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        predict_value = []
        label_value = []
        for id, (test_data, label) in enumerate(test_dataloader):
            predict = model(test_data)
            predict_value.append(predict.squeeze().detach().numpy())
            label_value.append(label.squeeze().detach().numpy())

        predict_value = np.array(predict_value)
        re_predict_value = predict_value * (data_case_max - data_case_min) + data_case_min

        new_day_data = np.zeros((1, InputSize))
        new_day_data[0,:R] = basic_reproduction_number[val_split_point+test_cycle+4*(day+1)]
        new_day_data[0,R+V+MOB*0:R+V+MOB*1] = grocery_and_pharmacy_percent_change_from_baseline[val_split_point+test_cycle+4*(day+1)]
        new_day_data[0,R+V+MOB*1:R+V+MOB*2] = transit_stations_percent_change_from_baseline[val_split_point+test_cycle+4*(day+1)]
        new_day_data[0,R+V+MOB*nb_MOB:R+V+MOB*nb_MOB+M-4] = test_data[0,N+R+V+MOB*nb_MOB:]
        new_day_data[0,-4:] = predict_value[0]
        test_data = torch.DoubleTensor(new_day_data)

        for count in range(0,N):
            days_predict.append(re_predict_value[0][count])     
            days_label.append(data[val_split_point+day*N+test_cycle, InputSize:][count] * (data_case_max - data_case_min) + data_case_min) 

    whole_predict.append(days_predict)
    whole_label.append(days_label)

for day in range(0,15):
    day_predict=[item[day] for item in whole_predict]
    day_predict = np.array(day_predict)
    day_label=[item[day] for item in whole_label]
    day_label = np.array(day_label)
    day_error_rate = np.mean(abs(day_predict - day_label) / (day_label))
    accuracy = (1 - day_error_rate)*100
    print(accuracy)