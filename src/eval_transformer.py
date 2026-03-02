from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
from typing import Any
from transformers import pipeline


def evaluate_transformer(model_name: str, data_loader: DataLoader, tokenizer: Any, max_length: int, device: str = 'cpu') -> tuple[float, float]:
    # generator_device = 0 if device == 'cuda' else -1
    generator_device = -1
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

    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=all_preds, references=all_refs)
    return results['rouge1'], results['rouge2']

