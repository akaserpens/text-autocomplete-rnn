import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from typing import Iterable, Iterator, Literal, List, Any

class NextTokenDataset(Dataset):
    r"""Dataset class

    Contains dict{``head``: tensor, ``tail``: tensor}.
    ``head`` is input (prompt). It has variable length.
    ``tail`` is target output. It has fixed length of ``max_output_length``.
    All heads and tails are wrapped in ``bos_token_id`` and ``eos_token_id``.

    Args:
        texts: actual texts
        tokenizer: used to convert strings to tokens
        bos_token_id: ``begin of sentence`` token id
        eos_token_id: ``end of sequence`` token id
        pad_token_id: ``padding`` token id
        max_input_length (int, optional): maximum input (prompt) length. Default is 50
        max_output_length (int, optional): maximum output (expected result) length. Default is 50
        split_mode (str, optional): how to generate samples from string, ``all`` | ``quarter``
            - ``all``: create all possible pairs of prompt and target from each string
            - ``quarter``: 3/4 of string as prompt, 1/4 as target
            Default is ``all``
    """
    def __init__(
            self,
            texts: Iterable[str],
            tokenizer: Any,
            bos_token_id: int,
            eos_token_id: int,
            pad_token_id: int,
            max_input_length: int = 50,
            max_output_length: int = 50,
            split_mode: Literal['all', 'quarter'] = 'all'
        ):

        self.samples = []

        # из максимальной длины вычитаем служебные символы (начало и конец предложения)
        max_input_length -= 2
        max_output_length -= 2

        for line in tqdm(texts):
            token_ids = tokenizer(line, add_special_tokens=False)['input_ids']

            if len(token_ids) < 2:
                continue

            if split_mode == 'all':
                split_fn = self._split_all
            elif split_mode == 'quarter':
                split_fn = self._split_quarter
            else:
                raise ValueError(f'Invalid split_mode: {split_mode}')

            for head, tail in split_fn(token_ids, max_input_length, max_output_length):
                output_padding = [pad_token_id] * (max_output_length - len(tail))
                self.samples.append((
                    [bos_token_id] + head + [eos_token_id],
                    [bos_token_id] + tail + [eos_token_id] + output_padding,
                ))

    def _split_quarter(self, token_ids: List[int], max_input_length: int, max_output_length: int) -> Iterator[tuple[List[int], List[int]]]:
        head_len = int(len(token_ids) * 0.75)
        head_len = min(head_len, max_input_length)
        head = token_ids[:head_len]
        tail = token_ids[head_len:head_len + max_output_length]
        yield (head, tail)

    def _split_all(self, token_ids: List[int], max_input_length: int, max_output_length: int) -> Iterator[tuple[List[int], List[int]]]:
        for i in range(1, len(token_ids) + 1):
            start_pos = max(0, i - max_input_length)
            head = token_ids[start_pos : i]
            tail = token_ids[i : i + max_output_length]
            yield (head, tail)
        
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

    return {
        'heads': padded_heads,
        'tails': torch.stack(tails, dim=0),
    }

def next_token_data_loader(dataset, batch_size, shuffle):
    r"""Creates DataLoader for NextTokenDataset. Pad all inputs to equal length. 
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_pad_batch)