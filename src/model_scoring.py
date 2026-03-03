from typing import Any

import evaluate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline


def score_model(model: nn.Module, data_loader: DataLoader, tokenizer: Any, max_new_tokens: int, device: str = 'cpu') -> tuple[float, float]:
    all_preds = []
    all_refs = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            reference = tokenizer.batch_decode(batch['tails'], skip_special_tokens=True)

            generated_tokens = model.generate(batch['heads'].to(device), max_new_tokens=max_new_tokens)
            generated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            all_preds += generated
            all_refs += reference

    return _calculate_score(predictions=all_preds, references=all_refs)

def score_transformer(model_name: str, data_loader: DataLoader, tokenizer: Any, max_length: int) -> tuple[float, float]:
    generator = pipeline('text-generation', model=model_name)

    all_preds = []
    all_refs = []

    for batch in tqdm(data_loader):
        prompts = tokenizer.batch_decode(batch['heads'], skip_special_tokens=True)
        reference = tokenizer.batch_decode(batch['tails'], skip_special_tokens=True)
        
        output = generator(prompts, max_new_tokens=max_length, pad_token_id=0)
        generated = [line[0]['generated_text'].strip() for line in output]

        all_preds += generated
        all_refs += reference

    return _calculate_score(predictions=all_preds, references=all_refs)

def _calculate_score(predictions, references):
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=references)
    return results['rouge1'], results['rouge2']