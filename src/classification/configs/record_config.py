from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RecordConfig:
    _training_name = "cifar_test"  # temp variable used for convenience.

    # Recording part
    RECORD_START = 0  # Checkpoints and TensorBoard are not recorded before this epoch.

    USE_CHECKPOINTS: bool = 1
    CHECKPOINTS_DIR: Path = Path("checkpoints", _training_name)  # Can be any path.
    CHECKPT_SAVE_FREQ: int = 10  # How often to save checkpoints (if they are better than the previous one)

    USE_TB: bool = 1
    TB_DIR: Path = Path("logs", _training_name)
    VAL_FREQ: int = 10  # How often to compute accuracy and images (also used for validation freq)
