import logging
import os

import wandb


def setup_logger(level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)

    rank = int(os.environ.get("RANK", 0))

    formatter = logging.Formatter(f"rank {rank} - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


logger = setup_logger()


class DynamicTable:
    def __init__(self, name: str):
        self.name = name
        self.table = None

    def add_data(
        self,
        **kwargs,
    ):
        if self.table is None:
            self.table = wandb.Table(columns=list(kwargs.keys()), log_mode="MUTABLE")
        self.table.add_data(
            *[kwargs[col] for col in self.table.columns],
        )

    def log(self):
        wandb.log({self.name: self.table})
