import tempdatahand as DataHandling
import torch
import matplotlib.pyplot as plt

dataset = DataHandling.dataset
device = "cuda"

lstm = torch.jit.load("LSTM.pt").to("cuda")
lstm.eval()
yarr = []
yhatarr = []
for data in dataset:
    y = data[1].to(device)
    data = data[0].to(device)
    data = data.reshape(1, data.size()[0], 1)
    yhat = lstm(data)[0][0]
    yarr.append(y.detach().cpu())
    yhatarr.append(yhat.detach().cpu())

plt.plot(range(len(yarr)), yarr)
plt.plot(range(len(yarr)), yhatarr, label = "yhatarr")
plt.legend()
plt.show()
