
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.models.prot_bert_encoder import ProtBERTEncoder


DATA_PATH = "dataset/kiba_processed.csv"
OUTPUT_PATH = "dataset/data_cache/protein_embeddings.npy"

df = pd.read_csv(DATA_PATH)


def preprocess_sequence(seq: str) -> str:
    return " ".join(list(seq))

df["fasta_processed"] = df["fasta"].apply(preprocess_sequence)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = ProtBERTEncoder().to(device)
encoder.model.eval()


protein_embeddings = {}

for prot_id, seq in tqdm(zip(df["prot_id"], df["fasta_processed"]), total=len(df)):
    if prot_id not in protein_embeddings:
        emb = encoder([seq])  
        protein_embeddings[prot_id] = emb.squeeze(0).cpu().numpy()


os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
np.save(OUTPUT_PATH, protein_embeddings)

print(f"✅ Saved {len(protein_embeddings)} protein embeddings to {OUTPUT_PATH}")
