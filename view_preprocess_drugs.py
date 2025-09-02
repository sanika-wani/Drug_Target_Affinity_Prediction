import os
import torch
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr  

CACHE_DIR = "dataset/data_cache/drug_graphs/"


torch.serialization.add_safe_globals([Data, DataEdgeAttr])

files = os.listdir(CACHE_DIR)
print("Cached files:", files[:5])


graph = torch.load(os.path.join(CACHE_DIR, files[0]), weights_only=False)

print("\nGraph summary:", graph)
print("Node features shape:", graph.x.shape)
print("Edge index shape:", graph.edge_index.shape)
print("Edge attributes shape:", graph.edge_attr.shape)

print("\nFirst 5 atoms (features):\n", graph.x[:5])
print("First 10 edges:\n", graph.edge_index[:, :10])
print("First 10 edge attrs:\n", graph.edge_attr[:10])
