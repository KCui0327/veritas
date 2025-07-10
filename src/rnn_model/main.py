import torch
import torch.nn as nn
import torch.optim as optim

from src.rnn_model.model import RNNModel
from src.rnn_model.trainer import train_model
from src.rnn_model.training_config import TrainingConfig

if __name__ == "__main__":
    # TODO: use dataloader

    model = RNNModel()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    training_config = TrainingConfig(
        train_dataloader=None,
        val_dataloader=None,
        save_dir="checkpoints",
        log_interval=10,
        eval_interval=1,
        epochs=100,
        optimizer=optimizer,
        loss_function=nn.CrossEntropyLoss(),
        use_cuda=torch.cuda.is_available(),
    )

    trainer.train_model(model, training_config)
