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
V = 20
MOB = 20
nb_MOB = 6 

nb_data = 370 - (M + N)
InputSize = M + R + V + MOB*nb_MOB
OutputSize = N

#new_cases
file = pd.read_csv('data_2021-Mar-07.csv')
file = file.iloc[::-1]
data_list = np.array(file['newCasesBySpecimenDate'])
data_list = data_list[31:]

norm = (data_list - data_list.min()) / (data_list.max() - data_list.min())
print('new_cases')
print(data_list.min())
print(data_list.max())
data_set = np.zeros((nb_data, M + N))

for row in range(nb_data):
    data_set[row] = norm[row:row + M + N] 


with open('new_cases.csv', 'wb') as f:
    np.savetxt(f, data_set, delimiter=",")

#basic_reproduction_number
data = pd.read_csv('owid-covid-data.csv')
data_list = np.array(data['reproduction_rate'])
data_list = data_list[68995-R:69357]
where_are_nan = np.isnan(data_list)
data_list[where_are_nan] = 1.6

norm = (data_list - data_list.min()) / (data_list.max() - data_list.min())
print('basic_reproduction_number')
print(data_list.min())
print(data_list.max())
data_set = np.zeros((nb_data, R))

for row in range(nb_data):
    data_set[row] = norm[row:row + R]             

with open('basic_reproduction_number.csv', 'wb') as f:
    np.savetxt(f, data_set, delimiter=",")

#people_vaccinated
data = pd.read_csv('owid-covid-data.csv')
data_list = np.array(data['people_vaccinated'])
data_list = data_list[68995-V:69357]
where_are_nan = np.isnan(data_list)
data_list[where_are_nan] = 0
data_clean  = np.zeros(362+V)
data_clean[307+V:] = data_list[307+V:]

norm = (data_clean - data_clean.min()) / (data_clean.max() - data_clean.min())
print('people_vaccinated')
print(data_clean.min())
print(data_clean.max())
data_set = np.zeros((nb_data, V))

for row in range(nb_data):
    data_set[row] = norm[row:row + V]  

with open('people_vaccinated.csv', 'wb') as f:
    np.savetxt(f, data_set, delimiter=",")

#retail_and_recreation_percent_change_from_baseline
data = pd.read_csv('2020_GB_Region_Mobility_Report.csv')
data_list = np.array(data['retail_and_recreation_percent_change_from_baseline'])
data_list = data_list[23-MOB:396]

norm = (data_list - data_list.min()) / (data_list.max() - data_list.min())
print('retail_and_recreation_percent_change_from_baseline')
print(data_list.min())
print(data_list.max())
data_set = np.zeros((nb_data, MOB))

for row in range(nb_data):
    data_set[row] = norm[row:row + MOB]    

with open('retail_and_recreation_percent_change_from_baseline.csv', 'wb') as f:
    np.savetxt(f, data_set, delimiter=",")

#grocery_and_pharmacy_percent_change_from_baseline
data = pd.read_csv('2020_GB_Region_Mobility_Report.csv')
data_list = np.array(data['grocery_and_pharmacy_percent_change_from_baseline'])
data_list = data_list[23-MOB:396]

norm = (data_list - data_list.min()) / (data_list.max() - data_list.min())
print('grocery_and_pharmacy_percent_change_from_baseline')
print(data_list.min())
print(data_list.max())
data_set = np.zeros((nb_data, MOB))

for row in range(nb_data):
    data_set[row] = norm[row:row + MOB]    

with open('grocery_and_pharmacy_percent_change_from_baseline.csv', 'wb') as f:
    np.savetxt(f, data_set, delimiter=",")

#grocery_and_pharmacy_percent_change_from_baseline
data = pd.read_csv('2020_GB_Region_Mobility_Report.csv')
data_list = np.array(data['grocery_and_pharmacy_percent_change_from_baseline'])
data_list = data_list[23-MOB:396]

norm = (data_list - data_list.min()) / (data_list.max() - data_list.min())
print('grocery_and_pharmacy_percent_change_from_baseline')
print(data_list.min())
print(data_list.max())
data_set = np.zeros((nb_data, MOB))

for row in range(nb_data):
    data_set[row] = norm[row:row + MOB]  

with open('grocery_and_pharmacy_percent_change_from_baseline.csv', 'wb') as f:
    np.savetxt(f, data_set, delimiter=",")

#parks_percent_change_from_baseline
data = pd.read_csv('2020_GB_Region_Mobility_Report.csv')
data_list = np.array(data['parks_percent_change_from_baseline'])
data_list = data_list[23-MOB:396]

norm = (data_list - data_list.min()) / (data_list.max() - data_list.min())
print('parks_percent_change_from_baseline')
print(data_list.min())
print(data_list.max())
data_set = np.zeros((nb_data, MOB))

for row in range(nb_data):
    data_set[row] = norm[row:row + MOB] 

with open('parks_percent_change_from_baseline.csv', 'wb') as f:
    np.savetxt(f, data_set, delimiter=",")

#transit_stations_percent_change_from_baseline
data = pd.read_csv('2020_GB_Region_Mobility_Report.csv')
data_list = np.array(data['transit_stations_percent_change_from_baseline'])
data_list = data_list[23-MOB:396]

norm = (data_list - data_list.min()) / (data_list.max() - data_list.min())
print('transit_stations_percent_change_from_baseline')
print(data_list.min())
print(data_list.max())
data_set = np.zeros((nb_data, MOB))

for row in range(nb_data):
    data_set[row] = norm[row:row + MOB] 

with open('transit_stations_percent_change_from_baseline.csv', 'wb') as f:
    np.savetxt(f, data_set, delimiter=",")

#workplaces_percent_change_from_baseline
data = pd.read_csv('2020_GB_Region_Mobility_Report.csv')
data_list = np.array(data['workplaces_percent_change_from_baseline'])
data_list = data_list[23-MOB:396]

norm = (data_list - data_list.min()) / (data_list.max() - data_list.min())
print('workplaces_percent_change_from_baseline')
print(data_list.min())
print(data_list.max())
data_set = np.zeros((nb_data, MOB))

for row in range(nb_data):
    data_set[row] = norm[row:row + MOB] 

with open('workplaces_percent_change_from_baseline.csv', 'wb') as f:
    np.savetxt(f, data_set, delimiter=",")

#residential_percent_change_from_baseline
data = pd.read_csv('2020_GB_Region_Mobility_Report.csv')
data_list = np.array(data['residential_percent_change_from_baseline'])
data_list = data_list[23-MOB:396]

norm = (data_list - data_list.min()) / (data_list.max() - data_list.min())
print('residential_percent_change_from_baseline')
print(data_list.min())
print(data_list.max())
data_set = np.zeros((nb_data, MOB))

for row in range(nb_data):
    data_set[row] = norm[row:row + MOB] 

with open('residential_percent_change_from_baseline.csv', 'wb') as f:
    np.savetxt(f, data_set, delimiter=",")