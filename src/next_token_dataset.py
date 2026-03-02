import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

class NextTokenDataset(Dataset):
    def __init__(self, texts, tokenizer, bos_token_id, eos_token_id, max_length):
        self.samples = []
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_length = max_length
        for line in tqdm(texts):
            token_ids = tokenizer(line, add_special_tokens=False)['input_ids']
            if len(token_ids) < 2:
                continue
            upper_bound = min(len(token_ids), max_length - 2) # минус два - потому что добавляем два служебных токена
            for i in range(1, upper_bound + 1):
                self.samples.append((
                    [self.bos_token_id] + token_ids[:i] + [self.eos_token_id],
                    [self.bos_token_id] + token_ids[i:] + [self.eos_token_id]
                ))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        head, tail = self.samples[index]
        return {
            'head': torch.tensor(head),
            'tail': torch.tensor(tail),
        }
    
def _pad_batch(batch):
    heads = [item['head'] for item in batch]
    tails = [item['tail'] for item in batch]
    padded_heads = pad_sequence(heads, batch_first=True)
    padded_tails = pad_sequence(tails, batch_first=True)

    return {
        'heads': padded_heads,
        'tails': padded_tails,
    }

def next_token_data_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_pad_batch)