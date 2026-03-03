import torch
import torch.nn as nn
from typing import Any
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm.auto import tqdm

class ModelTrainer:
    def train(
            self,
            input_ids: torch.Tensor,
            y_true: torch.Tensor,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: Optimizer,
            **kwargs
        ) :
        pass

    def validate(
            self,
            input_ids: torch.Tensor,
            y_true: torch.Tensor,
            model: nn.Module,
            criterion: nn.Module,
            **kwargs
        ) :
        pass

class LSTMTrainer(ModelTrainer):
    r"""
    Trains NextTokenLSTMPredictor
    """
    def train(
            self,
            input_ids: torch.Tensor,
            y_true: torch.Tensor,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: Optimizer
        ) :

        model.train()
        optimizer.zero_grad()
        out = model(input_ids)
        loss = self._calc_loss(criterion, out, y_true)
        loss.backward()
        optimizer.step()
        return loss.item()

    def validate(
            self,
            input_ids: torch.Tensor,
            y_true: torch.Tensor,
            model: nn.Module,
            criterion: nn.Module
        ) :

        out = model(input_ids)
        loss = self._calc_loss(criterion, out, y_true)
        return loss.item()


    def _calc_loss(self, criterion: Any, model_output: torch.Tensor, y_true: torch.Tensor):
        # модель предсказывает следующий токен, поэтому в качестве таргета берем только первый токен из датасета (пропускаем BOS)
        return criterion(model_output, y_true[:, 1])
    

class Seq2SeqTrainer(ModelTrainer):
    r"""
    Trains NextTokenSeq2SeqPredictor
    """
    def train(
            self,
            input_ids: torch.Tensor,
            y_true: torch.Tensor,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: Optimizer
        ) :

        model.train()
        optimizer.zero_grad()
        out = model(input_ids, max_new_tokens=y_true.size(1), target_ids=y_true)
        loss = self._calc_loss(criterion, out, y_true)
        loss.backward()
        optimizer.step()
        return loss.item()

    def validate(
            self,
            input_ids: torch.Tensor,
            y_true: torch.Tensor,
            model: nn.Module,
            criterion: nn.Module
        ) :

        out = model(input_ids, max_new_tokens=y_true.size(1))
        loss = self._calc_loss(criterion, out, y_true)
        return loss.item()


    def _calc_loss(self, criterion: Any, model_output: torch.Tensor, y_true: torch.Tensor):
        return criterion(
            model_output.view(-1, model_output.size(-1)),
            y_true.view(-1)
        )

def train_val_cycle(
        train_data: DataLoader,
        val_data: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        trainer: ModelTrainer,
        device: str = 'cpu'
    ) -> float:

    model.train()

    total_train_loss = 0
    for batch in tqdm(train_data):
        heads = batch['heads'].to(device)
        tails = batch['tails'].to(device)
        loss = trainer.train(heads, tails, model, criterion, optimizer)
        total_train_loss += loss
    avg_train_loss = total_train_loss / len(train_data)

    model.eval()

    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_data):
            heads = batch['heads'].to(device)
            tails = batch['tails'].to(device)
            loss = trainer.validate(heads, tails, model, criterion)
            total_val_loss += loss
    avg_val_loss = total_val_loss / len(val_data)

    return avg_train_loss, avg_val_loss