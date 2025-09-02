import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.data import Subset

from src.models.drug_gin_encoder import DrugGINEncoder
from src.models.fusion_module import CrossAttentionFusion
from dataset import DrugProteinDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")


drug_encoder = DrugGINEncoder(hidden_dim=256).to(device)
fusion = CrossAttentionFusion(drug_dim=256, protein_dim=1024, hidden_dim=512).to(device)


optimizer = torch.optim.Adam(
    list(drug_encoder.parameters()) + list(fusion.parameters()), 
    lr=1e-4, weight_decay=1e-5
)
criterion = nn.MSELoss()
scaler = GradScaler()  


dataset = DrugProteinDataset(
    csv_path="dataset/kiba_processed.csv",
    split_path="dataset/splits/cold_drug.json",  
    drug_cache_dir="dataset/data_cache/drug_graphs",
    protein_cache_path="dataset/data_cache/protein_embeddings.npy"
)


train_idx = [i for i in range(len(dataset)) if dataset.split_map[dataset.indices[i]] == "train"]
val_idx   = [i for i in range(len(dataset)) if dataset.split_map[dataset.indices[i]] == "val"]
test_idx  = [i for i in range(len(dataset)) if dataset.split_map[dataset.indices[i]] == "test"]


train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=16, shuffle=False)
test_loader  = DataLoader(Subset(dataset, test_idx), batch_size=16, shuffle=False)


def evaluate(loader):
    drug_encoder.eval()
    fusion.eval()
    preds, labels = [], []

    with torch.no_grad():
        for drug_graph, protein_emb, affinity, split in loader:
            drug_graph = drug_graph.to(device)
            protein_emb = protein_emb.to(device)
            affinity = affinity.to(device)

            
            drug_emb, _ = drug_encoder(drug_graph)  
            pred = fusion(drug_emb, protein_emb)

            preds.append(pred.squeeze().cpu())
            labels.append(affinity.cpu())

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    rmse = torch.sqrt(((preds - labels) ** 2).mean()).item()
    return rmse


epochs = 20
for epoch in range(epochs):
    drug_encoder.train()
    fusion.train()
    total_loss = 0

    for drug_graph, protein_emb, affinity, split in train_loader:
        drug_graph = drug_graph.to(device)
        protein_emb = protein_emb.to(device)
        affinity = affinity.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type):
            drug_emb, _ = drug_encoder(drug_graph)
            pred = fusion(drug_emb, protein_emb)
            loss = criterion(pred.squeeze(), affinity)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    val_rmse = evaluate(val_loader)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val RMSE: {val_rmse:.4f}")


test_rmse = evaluate(test_loader)
print(f"Final Test RMSE: {test_rmse:.4f}")
