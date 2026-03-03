import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

class NextTokenLSTMPredictor(nn.Module):
    r"""
    Use LSTM to predict next token in sequence.
    """
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        lengths = (input_ids != 0).sum(axis=1)

        emb = self.embedding(input_ids)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (last_hidden_state, _) = self.rnn(packed)
        
        linear_out = self.fc(last_hidden_state[-1])

        return linear_out
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        r"""
        Recursively predicts next token in sequence until ``max_new_tokens`` is predicted
        """
        result = []

        for _ in range(max_new_tokens):
            output = self(input_ids)
            predicted = output.argmax(dim=1).view(-1, 1)
            input_ids = torch.cat((input_ids, predicted), dim=1).to(input_ids.device)
            result.append(predicted)

        return torch.cat(result, dim=1)
    
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

def lstm_train_epoch(
        data_loader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: str = 'cpu'
    ) -> float:

    total_epoch_loss = 0
    for batch in tqdm(data_loader):
        heads = batch['heads'].to(device)
        tails = batch['tails'].to(device)
        # модель предсказывает следующий токен, поэтому в качестве таргета берем только первый токен из датасета (пропускаем BOS)
        loss = lstm_train(heads, tails[:, 1], model, criterion, optimizer)
        total_epoch_loss += loss
    return total_epoch_loss / len(data_loader)
