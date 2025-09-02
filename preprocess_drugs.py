import os
import hashlib
import torch
from torch_geometric.data import Data
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm


CACHE_DIR = "dataset/data_cache/drug_graphs/"
os.makedirs(CACHE_DIR, exist_ok=True)



def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),                                
        atom.GetTotalDegree(),                              
        atom.GetExplicitValence(),
        atom.GetFormalCharge(),
        atom.GetHybridization(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetChiralTag(),
        atom.GetTotalNumHs(),
        round(atom.GetMass() * 0.01, 2)                     
    ], dtype=torch.float)



def bond_features(bond):
    bt = bond.GetBondType()
    bond_type = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
    ]
    return torch.tensor([
        *bond_type,
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        int(bond.GetStereo())
    ], dtype=torch.float)



def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])

    
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        e = bond_features(bond)

        
        edge_index.append((i, j))
        edge_index.append((j, i))
        edge_attr.append(e)
        edge_attr.append(e)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr) if len(edge_attr) > 0 else torch.zeros((0, 7))

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data



def get_graph(smiles):
    
    h = hashlib.md5(smiles.encode()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{h}.pt")

    if os.path.exists(cache_file):
        return torch.load(cache_file)

    data = mol_to_graph(smiles)
    if data is not None:
        torch.save(data, cache_file)
    return data


if __name__ == "__main__":
    df = pd.read_csv("dataset/kiba_processed.csv")
    unique_smiles = df["smiles"].unique()

    print(f"Processing {len(unique_smiles)} unique drugs...")

    for smi in tqdm(unique_smiles):
        graph = get_graph(smi)  

    print("✅ Finished caching drug graphs in:", CACHE_DIR)
