import torch
from torch_geometric.data import Data

def build_graph(x, y):
    batch_size = 5000
    top_k = 15  # Connect to top 15 most similar profiles to prevent memory explosion
    
    # L2 Normalization so cosine similarity is just the dot product
    x_normalized = torch.nn.functional.normalize(x, p=2, dim=1)
    
    edge_sources = []
    edge_targets = []
    
    print(f"Constructing user graph via Top-{top_k} Cosine Similarity...")
    for i in range(0, x.size(0), batch_size):
        end_i = min(i + batch_size, x.size(0))
        x_batch = x_normalized[i:end_i]
        
        # Calculate similarity across the batch and the entire normalized feature set
        sim_matrix = torch.mm(x_batch, x_normalized.t())
        
        # For each node in the batch, get the top_k + 1 highest similarity scores 
        # (we add 1 because the node itself will be the highest match and we want to remove self-loops)
        _, top_indices = torch.topk(sim_matrix, k=top_k + 1, dim=1)
        
        # Create source indices
        src_indices = torch.arange(i, end_i).unsqueeze(1).expand_as(top_indices)
        
        src_flat = src_indices.reshape(-1)
        dst_flat = top_indices.reshape(-1)
        
        # Remove selfloops
        valid_edges_mask = src_flat != dst_flat
        
        edge_sources.append(src_flat[valid_edges_mask])
        edge_targets.append(dst_flat[valid_edges_mask])
        
    if len(edge_sources) > 0:
        source_nodes = torch.cat(edge_sources)
        target_nodes = torch.cat(edge_targets)
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    # Build complete PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data
