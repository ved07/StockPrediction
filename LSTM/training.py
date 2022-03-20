import torch
import DataHandling
import networks
from random import shuffle
# SETTING CONSTANTS
device = torch.device('cuda')
NUM_EPOCHS = 100  # number of epochs to train on

model = networks.LSTM(input_dim=1, hidden_dim=64, output_dim=1, num_layers=2).to(device)
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001)
dataset = DataHandling.dataset


def criterion(output, target): return abs(output-target).to(device)


losses = []
for epoch in range(NUM_EPOCHS):
    shuffle(dataset)
    for data in dataset:

        optimizer.zero_grad()
        y = data[1].to(device)
        data = data[0].to(device)
        data = data.reshape(1, data.size()[0], 1)
        yhat = model(data)[0][0]
        loss = criterion(yhat, y).to(device)
        loss.backward()
        optimizer.step()
        print(y, yhat)
        losses.append(loss)

    print(epoch+1, "completed", print("mean loss:", sum(losses)/len(losses)))
    scriptedLSTM = torch.jit.script(model)
    scriptedLSTM.save("LSTM.pt")

