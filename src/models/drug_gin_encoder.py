import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool, Set2Set

class MLP(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class DrugGINEncoder(nn.Module):
    
    def __init__(
        self,
        node_feat_dim=10,
        edge_feat_dim=7,
        hidden_dim=256,
        n_layers=5,
        dropout=0.2,
        readout="sum",  
        set2set_steps=3,
        train_eps=True,
        virtual_node=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.readout_type = readout
        self.virtual_node = virtual_node
        self.n_layers = n_layers

        
        self.node_in = nn.Linear(node_feat_dim, hidden_dim)

        
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for layer_idx in range(n_layers):
            
            nn_update = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINEConv(
                nn_update,
                edge_dim=edge_feat_dim,
                train_eps=train_eps
            )
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        
        if self.virtual_node:
            self.virtualnode_embed = nn.Embedding(1, hidden_dim)
            nn.init.constant_(self.virtualnode_embed.weight, 0.0)
            self.vn_mlps = nn.ModuleList()
            for _ in range(n_layers - 1):
                self.vn_mlps.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                )

        
        if readout == "set2set":
            self.set2set = Set2Set(hidden_dim, processing_steps=set2set_steps)
            self.readout_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        elif readout == "mean_max":
            self.readout_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            self.readout_proj = nn.Identity()

        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if edge_attr is None:
            raise ValueError("GINEConv requires edge_attr; got None.")

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        B = int(batch.max().item()) + 1 if x.numel() > 0 else 1

        x = self.node_in(x)

        
        if self.virtual_node:
            v = self.virtualnode_embed.weight[0].unsqueeze(0).expand(B, -1)
        else:
            v = None

        
        for layer_idx, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if self.virtual_node:
                x = x + v[batch]

            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = self.act(x)
            x = self.dropout(x)

            if self.virtual_node and layer_idx < self.n_layers - 1:
                pooled = global_add_pool(x, batch)
                v = v + pooled
                v = self.vn_mlps[layer_idx](v)

        node_feats = x

        
        if self.readout_type == "set2set":
            pooled = self.set2set(x, batch)
            pooled = self.act(self.readout_proj(pooled))
        elif self.readout_type == "mean_max":
            mean = global_mean_pool(x, batch)
            mx = global_max_pool(x, batch)
            pooled = torch.cat([mean, mx], dim=-1)
            pooled = self.act(self.readout_proj(pooled))
        else:
            pooled = global_add_pool(x, batch)
            pooled = self.readout_proj(pooled)

        return pooled, node_feats
