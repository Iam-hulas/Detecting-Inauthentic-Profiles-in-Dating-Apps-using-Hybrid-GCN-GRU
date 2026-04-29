import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from data_processing import load_and_process_data
from graph_builder import build_graph
from models.hybrid_model import HybridModel

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load processing data
    # Creating a dummy dataset to test the code execution if it doesn't exist
    try:
        x, y, df = load_and_process_data("dataset.csv")
    except FileNotFoundError:
        print("dataset.csv not found. Auto-generating a dummy dataset.csv for testing...")
        import numpy as np
        num_users = 1000
        dummy_df = pd.DataFrame({
            'app_usage_time_min': np.random.randint(0, 500, num_users),
            'message_sent_count': np.random.randint(0, 300, num_users),
            'swipe_right_ratio': np.random.rand(num_users),
            'likes_received': np.random.randint(0, 100, num_users),
            'mutual_matches': np.random.randint(0, 50, num_users),
            'profile_pics_count': np.random.randint(0, 5, num_users),
            'bio_length': np.random.randint(0, 100, num_users),
            'emoji_usage_rate': np.random.rand(num_users)
        })
        dummy_df.to_csv("dataset.csv", index=False)
        x, y, df = load_and_process_data("dataset.csv")

    # 2. Build graph
    data = build_graph(x, y)
    
    # Extract temporal sequence for GRU. 
    # Must use these exact 5 features to create 5-step sequence per user.
    seq_cols = [
        'app_usage_time_min', 
        'message_sent_count', 
        'swipe_right_ratio', 
        'likes_received', 
        'mutual_matches'
    ]
    
    temporal_features = df[seq_cols].copy().fillna(0)
    
    # Standardize temporal steps
    temporal_features = (temporal_features - temporal_features.mean()) / (temporal_features.std() + 1e-8)
    
    # Reshape into (num_users, 5, 1)
    temporal_x = torch.tensor(temporal_features.values, dtype=torch.float).unsqueeze(-1)
    
    # Move to device
    data = data.to(device)
    temporal_x = temporal_x.to(device)
    
    # 3. Instantiate hybrid model
    gcn_in_channels = data.x.size(1)
    model = HybridModel(gcn_in_channels=gcn_in_channels, gru_input_size=1, num_classes=3).to(device)
    
    # 4. Loss and Optimizer
    
    # Calculate class weights to handle severe class imbalance
    class_counts = torch.bincount(data.y, minlength=3).float()
    # Replace zeros with a small number to avoid division by zero
    class_counts[class_counts == 0] = 1.0 
    weights = len(data.y) / (3.0 * class_counts)
    
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train Loop
    print("\n--- Starting Training (30 Epochs) ---")
    model.train()
    
    for epoch in range(1, 31):
        optimizer.zero_grad()
        
        # Forward pass hybrid model
        logits = model(data.x, data.edge_index, temporal_x)
        
        # Calculate loss
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == data.y).sum().item() / data.num_nodes
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch: {epoch:02d}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")
            
    print("\nTraining complete.")
    
    # Save the model and data context for evaluation phase
    torch.save(model.state_dict(), "hybrid_model.pth")
    torch.save((data, temporal_x), "processed_data.pt")
    print("Saved 'hybrid_model.pth' and 'processed_data.pt'")
    
if __name__ == "__main__":
    train()
