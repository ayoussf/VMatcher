import dataclasses
from .train import train_config
from .test import test_config

@dataclasses.dataclass(frozen=True)
class main_config:
    """
    Training options.
    Args:
        validate_training: Validate during training.
    """
    train: train_config
    test: test_config