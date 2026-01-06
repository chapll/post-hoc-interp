import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from infra.config_manager import JobConfig


@dataclass
class ModelConfig(ABC):
    """Abstract base class for model configurations."""

    @classmethod
    def get_preset(cls, preset: str, job_config: Optional[JobConfig] = None):
        """Factory method to create a config from a preset name."""
        if not hasattr(cls, "PREDEFINED_CONFIGS"):
             raise NotImplementedError(f"{cls.__name__} must define PREDEFINED_CONFIGS")
             
        # We access the subclass's dictionary directly
        configs_dict = getattr(cls, "PREDEFINED_CONFIGS")
        
        if preset not in configs_dict:
             raise ValueError(f"Preset '{preset}' not found in {cls.__name__}.")

        # Instantiate the subclass
        model_config = cls(**configs_dict[preset])

        if job_config is not None:
            model_config.update(job_config)

        return model_config

    @abstractmethod
    def update(self, job_config: JobConfig):
        """
        Map JobConfig fields to their specific attributes within the ModelConfig.
        """
        pass

    def __str__(self):
        return json.dumps(asdict(self), indent=4)


@dataclass
class HNetConfig(ModelConfig):
    name: str = "hnet"
    model_dim: int = 1024
    
    # Example specific fields
    num_layers: int = 12

    PREDEFINED_CONFIGS = {
        "default": {
            "model_dim": 1024,
            "num_layers": 12,
        }
    }

    def update(self, job_config: JobConfig):
        # Implement the specific mapping logic here
        pass