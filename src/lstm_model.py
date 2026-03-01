import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Optimizer

class NextTokenLSTMPredictor(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, bos_token_id: int, eos_token_id: int):
        super().__init__()
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, vocab_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lengths = (x != 0).sum(axis=1)

        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, (last_hidden_state, _) = self.rnn(packed)
        
        linear_out = self.fc(last_hidden_state[-1])

        return linear_out
    
    def generate(self, input_ids, max_len):
        result = []
        for _ in range(max_len):
            output = self(input_ids.unsqueeze(0))
            predicted = output.squeeze().argmax()
            input_ids = torch.cat((input_ids, predicted.unsqueeze(0)), dim=0)
            result.append(predicted)
            if (predicted == self.eos_token_id):
                break

        return result
    
def lstm_train(
        input_ids: torch.Tensor,
        y_true: torch.Tensor,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer
    ) -> float:

    model.train()
    optimizer.zero_grad()
    out = model(input_ids)
    loss = criterion(out, y_true)
    loss.backward()
    optimizer.step()
    return loss.item()