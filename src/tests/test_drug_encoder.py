import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from src.models.drug_gin_encoder import DrugGINEncoder
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage  

graph_path = "dataset/data_cache/drug_graphs/0a2f96b9aa300de403332c8977369e32.pt"


with torch.serialization.safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
    data = torch.load(graph_path)

print("Graph loaded:")
print(data)
print("x:", data.x.shape)
print("edge_index:", data.edge_index.shape)
print("edge_attr:", data.edge_attr.shape)


if data.edge_attr is None:
    data.edge_attr = torch.zeros((data.edge_index.size(1), 7))


model = DrugGINEncoder(
    node_feat_dim=data.x.size(-1),
    edge_feat_dim=data.edge_attr.size(-1),
    hidden_dim=256,
    n_layers=5,
    dropout=0.2,
    readout="sum",
    virtual_node=True,
)

model.eval()
with torch.no_grad():
    pooled, node_feats = model(data)

print("\n=== Encoder Output ===")
print("Graph embedding (pooled):", pooled.shape)
print("Node embeddings:", node_feats.shape)


loader = DataLoader([data, data, data, data], batch_size=4)
batch = next(iter(loader))

with torch.no_grad():
    pooled_b, node_feats_b = model(batch)

print("\n=== Batched run (4 graphs) ===")
print("Graph embeddings:", pooled_b.shape)
print("Node embeddings:", node_feats_b.shape)

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.models.drug_gin_encoder import DrugGINEncoder






model = DrugGINEncoder(
    node_feat_dim=data.x.size(1),
    edge_feat_dim=data.edge_attr.size(1),
    hidden_dim=256,
    n_layers=5,
    dropout=0.2,
    readout="sum",
    virtual_node=True
)


model.eval()

with torch.no_grad():
    graph_emb, node_feats = model(data)

print("Graph embedding:", graph_emb.shape)
print("Node embeddings:", node_feats.shape)


n_samples = node_feats.size(0)
perplexity = min(30, max(5, n_samples // 3))  


node_emb_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(
    node_feats.detach().numpy()
)


plt.figure(figsize=(6, 6))
plt.scatter(node_emb_2d[:, 0], node_emb_2d[:, 1], c='skyblue', s=60)
for i, coord in enumerate(node_emb_2d):
    plt.text(coord[0], coord[1], str(i), fontsize=9)
plt.title("Node embeddings (t-SNE)")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.show()


plt.figure(figsize=(4, 4))
plt.scatter(graph_emb[:, 0].numpy(), graph_emb[:, 1].numpy(), c='red', s=80)
plt.title("Graph embeddings")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()

import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


with torch.no_grad():
    graph_emb, node_feats = model(data)

print("Graph embedding shape:", graph_emb.shape)
print("Node embeddings shape:", node_feats.shape)



n_nodes = node_feats.size(0)
perplexity = min(5, n_nodes - 1)   

node_emb_2d = TSNE(
    n_components=2,
    perplexity=perplexity,
    random_state=42
).fit_transform(node_feats.detach().cpu().numpy())


edge_index = data.edge_index.cpu().numpy()
G = nx.Graph()
for i in range(data.x.size(0)):
    G.add_node(i)

for src, dst in edge_index.T:
    G.add_edge(int(src), int(dst))


pos = {i: node_emb_2d[i] for i in range(node_emb_2d.shape[0])}

plt.figure(figsize=(7, 7))
nx.draw(
    G, pos,
    with_labels=True,
    node_color="skyblue",
    node_size=600,
    font_size=9,
    edge_color="gray"
)
plt.title("Graph structure in GIN embedding space")


plt.savefig("graph_embedding.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Saved graph visualization to graph_embedding.png")
