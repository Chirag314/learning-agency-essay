
from __future__ import annotations
import pandas as pd
from typing import List, Optional
from transformers import AutoTokenizer
from torch.utils.data import Dataset

def detect_columns(df: pd.DataFrame, text_candidates: List[str], target_candidates: List[str], id_candidates: List[str]):
    text_col = next((c for c in text_candidates if c in df.columns), None)
    target_col = next((c for c in target_candidates if c in df.columns), None)
    id_col = next((c for c in id_candidates if c in df.columns), None)
    return text_col, target_col, id_col

class EssayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, text_col: str, target_col: Optional[str]=None, max_length: int=1024):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.target_col = target_col
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row[self.text_col]
        enc = self.tokenizer(
            str(text),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )
        item = {k: v for k, v in enc.items()}
        if self.target_col is not None and self.target_col in self.df.columns:
            item["labels"] = float(row[self.target_col])
        return item
