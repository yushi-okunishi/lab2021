import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_num=5, hidden_num=10, output_num=1):
        super().__init__()
        self.lin1 = nn.Linear(input_num, hidden_num)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_num, output_num)

    def forward(self, x):
        return self.lin2(self.relu(self.lin1(x)))