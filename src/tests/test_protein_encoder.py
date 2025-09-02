
import torch
import pandas as pd
from src.models.prot_bert_encoder import ProtBERTEncoder

def preprocess_sequence(seq: str) -> str:
    
    return " ".join(list(seq))

def test_protbert_on_dataset():
    
    df = pd.read_csv("dataset/kiba_processed.csv")
    
    
    sequences = df["fasta"].iloc[:5].apply(preprocess_sequence).tolist()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProtBERTEncoder().to(device)
    model.model.eval()

    
    embeddings = model(sequences)
    print("Input batch size:", len(sequences))
    print("Embedding shape:", embeddings.shape)  

if __name__ == "__main__":
    test_protbert_on_dataset()
