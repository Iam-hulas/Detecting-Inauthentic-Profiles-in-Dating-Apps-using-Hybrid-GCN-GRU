import torch
import torch.nn as nn
from models.gcn_model import GCNModel
from models.gru_model import GRUModel

class HybridModel(nn.Module):
    def __init__(self, gcn_in_channels, gru_input_size=1, gcn_hidden=64, gru_hidden=32, num_classes=3):
        super(HybridModel, self).__init__()
        
        self.gcn = GCNModel(in_channels=gcn_in_channels, hidden_channels=gcn_hidden)
        self.gru = GRUModel(input_size=gru_input_size, hidden_size=gru_hidden, num_layers=1)
        
        # Fully connected layer definition handling exactly 64 (GCN) + 32 (GRU)
        combined_size = gcn_hidden + gru_hidden
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, static_x, edge_index, temporal_x):
        # 1. Gather structural features
        gcn_emb = self.gcn(static_x, edge_index)
        
        # 2. Gather temporal sequences
        gru_emb = self.gru(temporal_x)
        
        # 3. Concatenate both embeddings
        combined_emb = torch.cat((gcn_emb, gru_emb), dim=1)
        
        # 4. Predict logits over the 3 target classes
        logits = self.fc(combined_emb)
        
        return logits
