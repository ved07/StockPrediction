# importing dependencies
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
# we will be using the nyse data for now
STOCK_EXCHANGE = "nyse"
WINDOW = 80

path = Path().parent.absolute().__str__()
path = path.split("\\")[:-1]

path.append("data")
path.append(STOCK_EXCHANGE)
path.append("csv")

path = "".join([item+"/" for item in path])

print(path)

training_data = []

counter = 0
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        data = pd.read_csv(path+filename)
        item = data["Open"].to_numpy()
        item = item/np.linalg.norm(item)
        item = torch.from_numpy(item)
        training_data.append(item)
        counter+=1
        if counter>2:
            break

dataset = []
for stock in training_data[1:]:
    for index in range(len(stock)-WINDOW-1):
        dataset.append((stock[index:index+WINDOW].type(torch.FloatTensor), stock[index+WINDOW].type(torch.FloatTensor)))

