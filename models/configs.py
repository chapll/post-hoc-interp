import json
from dataclasses import asdict, dataclass
from typing import List

from infra.config_manager import JobConfig


@dataclass
class ModelConfig:
    # TODO: Finish this
    model_dim: int
    