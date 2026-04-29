import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(GRUModel, self).__init__()
        # PyTorch GRU 
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
    def forward(self, x):
        # x input shape is strictly expected to be:
        # (batch_size, sequence_length=5, feature_dim=1)
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        
        # Returns final hidden state embedding for each user
        embedding = out[:, -1, :] 
        
        return embedding
