import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers = 1):
        super().__init__()
        self.hidden_size = hidden_dim
        self.LSTM = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_layers)
        self.num_layers = num_layers
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to("cuda")
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to("cuda")
        out, _ = self.LSTM(x, (h0.detach(), c0.detach()))
        out = out[:, -1, :]
        out = self.fc(out).to("cuda")
        return out

