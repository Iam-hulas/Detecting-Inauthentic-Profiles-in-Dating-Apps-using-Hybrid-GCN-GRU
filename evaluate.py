import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from models.hybrid_model import HybridModel

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load processed graph structure and temporal features from training session
    try:
        data, temporal_x = torch.load("processed_data.pt", weights_only=False)
        data = data.to(device)
        temporal_x = temporal_x.to(device)
    except FileNotFoundError:
        print("Error: 'processed_data.pt' not found. Please run train.py first to generate the necessary graph/data.")
        return

    gcn_in_channels = data.x.size(1)
    model = HybridModel(gcn_in_channels=gcn_in_channels, gru_input_size=1, num_classes=3).to(device)
    
    # Load model weights
    try:
        model.load_state_dict(torch.load("hybrid_model.pth", weights_only=True))
        model.eval()
    except FileNotFoundError:
        print("Error: 'hybrid_model.pth' not found. Please run train.py first to train the model weights.")
        return
        
    # Forward Pass Evaluation
    with torch.no_grad():
        logits = model(data.x, data.edge_index, temporal_x)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_true = data.y.cpu().numpy()
        
    print("\n==================================")
    print("       EVALUATION RESULTS         ")
    print("==================================")
    
    # 1. Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, preds))
    
    # 2. Classification Report
    print("\nClassification Report:")
    target_names = ['Authentic', 'Potentially Inauthentic', 'Inauthentic']
    
    # Extract only the target labels safely present in True distributions to prevent Sklearn errors
    unique_classes = sorted(list(set(y_true)))
    present_targets = [target_names[i] for i in unique_classes]
    
    print(classification_report(y_true, preds, labels=unique_classes, target_names=present_targets, zero_division=0))
    
    # 3. Global Accuracy
    acc = accuracy_score(y_true, preds)
    print(f"\nOverall Accuracy: {acc:.4f}\n")

if __name__ == "__main__":
    evaluate()
