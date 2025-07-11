import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloaders
from src.rnn_model.model import FakeNewsDetector
from src.rnn_model.trainer import train_model
from src.rnn_model.training_config import TrainingConfig

if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 20000
    embed_dim = 128
    hidden_dim = 128
    num_layers = 2
    dropout_rate = 0.5
    bidirectional = True
    output_dim = 1
    lr = 1e-3

    model = FakeNewsDetector(
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout_rate,
        bidirectional,
        output_dim,
    )

    train_dataloader, val_dataloader = get_dataloaders()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    training_config = TrainingConfig(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        save_dir="checkpoints",
        log_interval=10,
        eval_interval=1,
        epochs=100,
        optimizer=optimizer,
        loss_function=nn.BCELoss(),
        use_cuda=torch.cuda.is_available(),
    )

    train_model(model, training_config)
