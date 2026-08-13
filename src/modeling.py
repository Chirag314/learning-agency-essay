
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class DebertaRegressor(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        # Force fp32 master weights: this checkpoint is stored on the HF Hub in fp16, and
        # transformers now loads it in its stored dtype by default. Combined with this repo's
        # torch.cuda.amp.autocast + GradScaler training loop (src/train.py), fp16 master
        # weights make GradScaler raise "Attempting to unscale FP16 gradients" — GradScaler
        # requires fp32 master weights, with autocast handling the fp16 compute during the
        # forward pass. Found and fixed 2026-08-13.
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config, dtype=torch.float32)
        self.head = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        logits = self.head(cls)
        return {"logits": logits}
