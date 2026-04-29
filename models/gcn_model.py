import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super(GCNModel, self).__init__()
        # 2-layer GCN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Returns embedding vector
        return x
