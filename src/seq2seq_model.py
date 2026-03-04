import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class NextTokenGenerator(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, input_ids: torch.Tensor, max_new_tokens: int, target_ids: torch.Tensor | None = None) -> torch.Tensor:
        lengths = (input_ids != 0).sum(axis=1).cpu()

        all_outputs = []

        hidden = None

        for i in range(max_new_tokens):
            out, hidden = self._forward_step(input_ids, lengths if i == 0 else None, hidden)
            all_outputs.append(out)

            if target_ids is not None:
                input_ids = target_ids[:, i].unsqueeze(1) # teacher forcing
            else:
                input_ids = out.argmax(-1).reshape(-1, 1)

        return torch.stack(all_outputs, dim=1)
        
    
    def _forward_step(self, input_ids: torch.Tensor, lengths: torch.Tensor | None, prev_hidden_state: tuple[torch.Tensor, torch.Tensor] = None):
        emb = self.embedding(input_ids)
        if lengths is not None:
            emb = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden_state, cell_state) = self.rnn(emb, prev_hidden_state)
        
        linear_out = self.fc(hidden_state[-1])

        return linear_out, (hidden_state, cell_state)
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        r"""
        Generates ``max_new_tokens`` tokens based on ``input_ids``
        """
        output = self(input_ids, max_new_tokens)
        predicted = output.argmax(dim=-1)

        return predicted
