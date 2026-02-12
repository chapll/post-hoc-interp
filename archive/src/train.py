from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# Load environment variables from .env BEFORE Hydra runs
load_dotenv()

# Register custom resolvers for date/time in directory paths
OmegaConf.register_new_resolver("now_date", lambda: datetime.now().strftime("%Y%m%d"), replace=True)
OmegaConf.register_new_resolver("now_time", lambda: datetime.now().strftime("%H%M%S"), replace=True)

import hydra
import torch
import torch.nn as nn

from src.infra.logging import get_logger
from src.models.mlp import TwoLayerMLP

# Compute absolute path to configs directory
# __file__ points to src/train.py, configs is at ../configs relative to src/
CONFIG_PATH = str((Path(__file__).resolve().parent.parent / "configs").resolve())


def train(cfg: DictConfig, logger) -> None:
    """Train a simple MLP on the identity task (0→0, 1→1)."""
    torch.manual_seed(cfg.job.seed)

    # Create model
    model = TwoLayerMLP(input_dim=1, hidden_dim=16, output_dim=1)
    logger.write(f"Model: {model}")
    logger.write(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Simple optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)
    criterion = nn.MSELoss()

    # Training data: identity function (input=target)
    # 0 → 0, 1 → 1
    inputs = torch.tensor([[0.0], [1.0]])
    targets = torch.tensor([[0.0], [1.0]])

    # Training loop
    model.train()
    for step in range(cfg.training.steps):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == cfg.training.steps - 1:
            logger.write(f"Step {step}/{cfg.training.steps} | Loss: {loss.item():.6f}")

    # Test final outputs
    model.eval()
    with torch.no_grad():
        test_0 = model(torch.tensor([[0.0]]))
        test_1 = model(torch.tensor([[1.0]]))
        logger.write(f"Final results:")
        logger.write(f"  Input 0 -> Output {test_0.item():.4f} (expected: 0)")
        logger.write(f"  Input 1 -> Output {test_1.item():.4f} (expected: 1)")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    # Save resolved config to run directory
    run_dir = Path(cfg.deploy_paths.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    # Initialize logger (writes to both terminal and log.txt)
    logger = get_logger(cfg)

    logger.write("=" * 50)
    logger.write(f"Experiment: {cfg.job.exp_name}")
    logger.write(f"Run directory: {cfg.deploy_paths.run_dir}")
    logger.write("=" * 50)
    logger.write(OmegaConf.to_yaml(cfg), log_terminal=False)  # Config only to file
    logger.write("=" * 50)

    train(cfg, logger)

    logger.close()


if __name__ == "__main__":
    main()
