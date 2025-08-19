
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class DebertaRegressor(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.head = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        logits = self.head(cls)
        return {"logits": logits}
