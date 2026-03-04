from typing import Any, Iterable, Iterator, List, Literal

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class NextTokenDataset(Dataset):
    r"""Dataset class

    Contains dict{``head``: tensor, ``tail``: tensor}.
    ``head`` is input (prompt). It has variable length.
    ``tail`` is target output. It has fixed length of ``max_output_length``.
    All heads and tails are wrapped in ``bos_token_id`` and ``eos_token_id``.

    Args:
        texts: actual texts
        tokenizer: used to convert strings to tokens
        split_num (int, optional): number of pairs to generate from each text. Default is 3
            If ``split_num`` is set to 1, text will be split as 0.75/0.25.
            Otherwise, text will be split at ``split_num`` equaly distributed points.
        max_output_length (int, optional): maximum target length. Default is 5
    """
    def __init__(
            self,
            texts: Iterable[str],
            tokenizer: Any,
            split_num: int = 3,
            max_output_length: int = 5
        ):

        self.samples = []

        for line in tqdm(texts):
            token_ids = tokenizer(line, add_special_tokens=True)['input_ids']

            if len(token_ids) < 4:
                continue

            if split_num == 1:
                head_len = int(len(token_ids) * 0.75)
                edges = [head_len]
            else:
                # по краям будут добавлены токены начала и конца строки,
                # которые мы не хотим оставлять в одиночестве. Поэтому по -2 с каждой стороны
                edges = np.linspace(2, len(token_ids) - 2, split_num, dtype=int)
                edges = np.unique(edges)

            for edge in np.unique(edges):
                head, tail = token_ids[:edge], token_ids[edge : edge + max_output_length]
                self.samples.append((head, tail))
        
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

def next_token_data_loader(dataset: NextTokenDataset, batch_size: int, shuffle: bool):
    r"""Creates DataLoader for NextTokenDataset. Pad all inputs to equal length. 
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_pad_batch)