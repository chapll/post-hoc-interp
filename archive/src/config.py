from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JobConfig:
    exp_name: str = "default-job"
    seed: int = 13


@dataclass
class ModelConfig:
    name: str = "default"
    size: str = "5B"
    lr: float = 1e-4


@dataclass
class TrainingConfig:
    dataset_path: Optional[str] = None
    global_batch_size: int = 8
    grad_accum_steps: int = 1
    warmup_steps: int = 50
    steps: int = 5000
    gc_freq: int = 50
    adapter_method: Optional[str] = None


@dataclass
class CheckpointConfig:
    init_state_dir: Optional[str] = None
    interval: int = 0
    resume: bool = False
    resume_step: int = -1
    timeout_minutes: int = 0


@dataclass
class WandBConfig:
    disable: bool = False
    project: str = "cs224n"
    entity: str = "chapll"
    log_interval: int = 50
    alert: bool = False


@dataclass
class ParallelismConfig:
    tp_sharding: int = 1
    dp_sharding: int = 1
    dp_replicate: int = 1
    fsdp_unsharded_dtype: str = "bfloat16"


@dataclass
class CommConfig:
    init_timeout_seconds: int = 300


@dataclass
class DeployPaths:
    exp_dir: str = "./exp"
    data_dir: str = "./data"
    checkpoint_dir: str = "${deploy_paths.exp_dir}/${job.exp_name}/checkpoint"


@dataclass
class Config:
    job: JobConfig = field(default_factory=JobConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    deploy_paths: DeployPaths = field(default_factory=DeployPaths)


