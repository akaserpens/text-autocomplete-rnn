# Inspired by: https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

class NextTokenSeq2SeqPredictor(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, bos_token_id: int):
        r"""
        Encoder-Decoder for short text completion
        """
        super().__init__()
        self.bos_token_id = bos_token_id
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder_fc = nn.Linear(hidden_dim, vocab_size)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.decoder_fc.weight)
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, input_ids: torch.Tensor, max_new_tokens: int, target_ids: torch.Tensor | None = None) -> torch.Tensor:
        encoder_state = self._encode(input_ids)
        decoder_output = self._decode(input_ids.size(0), encoder_state, max_new_tokens=max_new_tokens, device=input_ids.device, target_ids=target_ids)
        return decoder_output
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        r"""
        Generates ``max_new_tokens`` tokens based on ``input_ids``
        """
        output = self(input_ids, max_new_tokens)
        predicted = output.argmax(dim=1)

        return predicted

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        lengths = (input_ids != 0).sum(axis=1).cpu()
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        return hidden
    
    def _decode(self, batch_size: int, encoder_hidden: torch.Tensor, max_new_tokens: int, device: str, target_ids: torch.Tensor | None = None):
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(self.bos_token_id)
        decoder_hidden = encoder_hidden
        all_outputs = []

        for i in range(max_new_tokens):
            decoder_output, decoder_hidden  = self._decode_step(decoder_input, decoder_hidden)
            all_outputs.append(decoder_output)

            if target_ids is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_ids[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(all_outputs, dim=1)
        return decoder_outputs

    def _decode_step(self, input_ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        output, hidden = self.decoder(embedded)
        linear_output = self.decoder_fc(output)
        return linear_output, hidden
