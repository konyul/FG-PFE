import torch
import torch.nn as nn

class SENet(nn.Module):
    def __init__(self, in_channel, reduction_ratio):
        super(SENet, self).__init__()
        self.globalAvgPool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_features=in_channel, out_features=round(in_channel/reduction_ratio))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=round(in_channel/reduction_ratio), out_features=in_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x[None,:]
        out = self.globalAvgPool(x.permute(0,2,1))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(1, 1, out.size(1))
        out = out * x
        return out.squeeze(0).contiguous()