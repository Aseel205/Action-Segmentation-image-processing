import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TemporalConvLayer, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.norm(out)
        return out

class SimpleTCN(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=4, hidden_dim=64):
        super(SimpleTCN, self).__init__()
        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            dilation = 2 ** i  # exponentially increasing dilation
            layers.append(TemporalConvLayer(in_channels, hidden_dim,
                                            kernel_size=3, dilation=dilation))
            in_channels = hidden_dim
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # x shape: (batch, feature_dim, seq_len)
        out = self.tcn(x)
        out = self.classifier(out)  # (batch, num_classes, seq_len)
        out = out.transpose(1, 2)   # (batch, seq_len, num_classes)
        return out
