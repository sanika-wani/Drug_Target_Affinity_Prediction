
import os
import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import hashlib


class DrugProteinDataset(Dataset):
    def __init__(self, csv_path, split_path, drug_cache_dir, protein_cache_path):
        
        self.df = pd.read_csv(csv_path)

        
        with open(split_path, "r") as f:
            split = json.load(f)
        self.indices = split["train"] + split["val"] + split["test"]

        
        self.split_map = {}
        for idx in split["train"]:
            self.split_map[idx] = "train"
        for idx in split["val"]:
            self.split_map[idx] = "val"
        for idx in split["test"]:
            self.split_map[idx] = "test"

        self.drug_cache_dir = drug_cache_dir

        
        self.protein_embeddings = np.load(protein_cache_path, allow_pickle=True).item()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]

        smiles = row["smiles"]       
        prot_id = row["prot_id"]
        affinity = torch.tensor(float(row["affinity"]), dtype=torch.float32)

        
        h = hashlib.md5(smiles.encode()).hexdigest()
        drug_path = os.path.join(self.drug_cache_dir, f"{h}.pt")

        if not os.path.exists(drug_path):
            raise FileNotFoundError(f"Graph not found for SMILES {smiles} at {drug_path}")
        drug_graph: Data = torch.load(drug_path, weights_only=False)

        
        if prot_id not in self.protein_embeddings:
            raise KeyError(f"Protein ID {prot_id} not found in cached ProtBERT embeddings.")
        protein_emb = torch.tensor(self.protein_embeddings[prot_id], dtype=torch.float32)

        return drug_graph, protein_emb, affinity, self.split_map[real_idx]
