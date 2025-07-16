from dataclasses import dataclass
from typing import Callable, Optional

import torch.optim as optim
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    """Configuration class for training hyperparameters."""

    train_dataloader: DataLoader = None
    val_dataloader: Optional[DataLoader] = None

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0

    optimizer: optim.Optimizer = None
    loss_function: Callable = None
    use_cuda: bool = False

    # Logging and saving
    save_dir: Optional[str] = None
    log_interval: int = 10  # log every n batches
    eval_interval: int = 1  # evaluate every n epochs

    def get_config_unique_name(self) -> str:
        """Get a unique name for the training configuration."""
        return f"{self.batch_size}_{self.learning_rate}_{self.weight_decay}"
