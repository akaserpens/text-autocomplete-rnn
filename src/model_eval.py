import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
from typing import Any

def evaluate_model(model: nn.Module, val_data_loader: DataLoader, tokenizer: Any, max_length: int, device: str = 'cpu') -> tuple[float, float]:
    all_preds = []
    all_refs = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_data_loader):
            generated = [model.generate(line.to(device), max_length=max_length) for line in batch['heads']]
            generated = [tokenizer.decode(line, skip_special_tokens=True) for line in generated]
            all_preds += generated
            reference = [tokenizer.decode(line, skip_special_tokens=True) for line in batch['tails']]
            all_refs += reference
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=all_preds, references=all_refs)
    return results['rouge1'], results['rouge2']