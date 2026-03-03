import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
from typing import Any

def evaluate_model(model: nn.Module, data_loader: DataLoader, tokenizer: Any, max_new_tokens: int, device: str = 'cpu') -> tuple[float, float]:
    all_preds = []
    all_refs = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            reference = tokenizer.batch_decode(batch['tails'], skip_special_tokens=True)

            generated_tokens = model.generate(batch['heads'].to(device), max_new_tokens=max_new_tokens)
            generated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            all_preds += generated
            all_refs += reference

    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=all_preds, references=all_refs)
    return results['rouge1'], results['rouge2']