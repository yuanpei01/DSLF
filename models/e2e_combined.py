import torch
import torch.nn as nn
from dataset_config import DATASET_CONFIG


class E2ECombined(nn.Module):
    def __init__(self, GLAN, Net, data):
        super(E2ECombined, self).__init__()
        cfg = DATASET_CONFIG[data]
        self.glan_model = GLAN
        self.net_model = Net
        self.fc1 = nn.Linear(856, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, cfg["num_classes"])
        self.relu = nn.ReLU()

    def forward(self, X_tid, X_source, X_replies, data):
        _, glan_output = self.glan_model(X_tid, X_source, X_replies)
        _, net_output = self.net_model(data)
        combined_features = torch.cat((glan_output, net_output), dim=1)
        x = self.relu(self.fc1(combined_features))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
