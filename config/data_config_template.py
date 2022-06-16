import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DataConfig:
    DATA_PATH: Path = Path("path", "to", "dataset")
    _training_name = "cifar_test"  # temp variable used for convenience.

    # Recording part
    RECORD_START = 0  # Checkpoints and TensorBoard are not recorded before this epoch

    USE_CHECKPOINTS: bool = True
    CHECKPOINTS_DIR: Path = Path("checkpoints", _training_name)  # Can be any path.
    CHECKPT_SAVE_FREQ: int = 10  # How often to save checkpoints (if they are better than the previous one)

    USE_TB: bool = True
    TB_DIR: Path = Path("logs", _training_name)
    VAL_FREQ: int = 10  # How often to compute accuracy and images (also used for validation freq)

    # Number of workers to use for dataloading
    NB_WORKERS: int = field(default_factory=lambda: int(os.cpu_count() * 0.8))

    # Build a map between id and class names  (TODO: should be done using typing + a field)
    LABEL_MAP = {}
    with open(DATA_PATH / "classes.names") as text_file:
        for key, line in enumerate(text_file):
            LABEL_MAP[key] = line.strip()
    NB_CLASSES: int = len(LABEL_MAP)


def get_data_config() -> DataConfig:
    return DataConfig()
