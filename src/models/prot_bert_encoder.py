
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List

class ProtBERTEncoder(nn.Module):
    

    def __init__(self, model_name: str = "Rostlab/prot_bert_bfd"):
        super(ProtBERTEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size  

    def forward(self, sequences: List[str]) -> torch.Tensor:
        
        
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        
        with torch.no_grad():
            outputs = self.model(**inputs)

        
        
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings
