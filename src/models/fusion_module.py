import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, drug_dim=256, protein_dim=1024, hidden_dim=512, num_heads=8):
        super(CrossAttentionFusion, self).__init__()

        
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.prot_proj = nn.Linear(protein_dim, hidden_dim)

        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, drug_emb, prot_emb):
        
        drug_h = self.drug_proj(drug_emb).unsqueeze(1)   
        prot_h = self.prot_proj(prot_emb).unsqueeze(1)   

        
        fused, _ = self.cross_attention(query=drug_h, key=prot_h, value=prot_h)  
        fused = fused.squeeze(1)  

        return self.mlp(fused)  
