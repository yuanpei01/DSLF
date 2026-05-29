import torch
import torch.nn as nn
from dataset_config import DATASET_CONFIG


class DLCombined(nn.Module):
    def __init__(self, data):
        super(DLCombined, self).__init__()
        cfg = DATASET_CONFIG[data]
        self.fc1 = nn.Linear(856, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, cfg["num_classes"])
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        combined_features = torch.cat((x1, x2), dim=1)
        x = self.relu(self.fc1(combined_features))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
