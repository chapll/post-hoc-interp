# Adapted from https://github.com/test-time-training/ttt-video-dit/blob/main/ttt/infra/config_manager.py
# Copyright 2025 test-time-training. All rights reserved.
# Licensed under the MIT License

import argparse
import json
import os
import os.path as osp
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, get_type_hints

from dotenv import load_dotenv

load_dotenv()


@dataclass
class JobExpConfig:
    """Job-level configs."""

    config_file: Optional[str] = field(default=None, metadata={"help": "Job config file"})
    exp_name: str = field(default="default-job", metadata={"help": "Description of the job"})
    dump_folder: str = field(
        default=os.getenv("EXP_DIR", os.path.join(os.getcwd(), "exp")),
        metadata={"help": "Location to dump logs of the job"},
    )
    seed: int = field(default=13, metadata={"help": "Random seed for the job"})


@dataclass
class TrainingConfig:
    """Training configuration options."""

    dataset_path = field(default=None, metadata={"help": "Path to the dataset in the file system."})
    global_batch_size = field(default=8, metadata={"help": "Global batch size."})
    grad_accum_steps = field(default=1, metadata={"help": "Grad accumulatation steps."})
    warmup_steps = field(default=50, metadata={"help": "The number of steps for lr scheduler warmup."})
    steps = field(default=5000, metadata={"help": "How many train steps to run"})
    gc_freq = field(default=50, metadata={"help": "Python garbage control scheduling interval, in steps"})


@dataclass
class EvalConfig:
    """Evaluation-specific configuration options."""

    output_dir: str = field(default="./output", metadata={"help": "Directory to save generated results"})


@dataclass
class WandBConfig:
    


def string_list(input_str: str) -> List[str]:
    """Parse a comma-separated string into a list of strings."""
    return input_str.split(",")


class JobConfig:
    """
    A helper class to manage the train configuration.
    Semantics:
    - Default config is loaded from a toml file. If no toml file is provided,
    then the default config is loaded from argparse defaults.
    - if toml file has missing keys, they are filled with argparse defaults.
    - if additional explicit cmd args are provided in addition to the toml
    file, they will override the toml config and the argparse defaults

    precedence order: cmdline > toml > argparse default

    Arg parsing semantics:

    Each argument starts with <prefix>_ which is the config name in the toml file
    followed by name of the option in the toml file. For ex,
    model.name translates to:
        [model]
        name
    in the toml file
    """

    # Define class-level attributes for editor intellisense
    job: JobExpConfig
    training: TrainingConfig
    eval: EvalConfig

    # Map config names to their corresponding dataclass types
    _config_types: Dict[str, Type] = {
        "job": JobExpConfig,
        "training": TrainingConfig,
    }

    def __init__(self, eval_mode=False):
        # Initialize config attributes with default values
        for config_name, config_class in self._config_types.items():
            setattr(self, config_name, config_class())

        # Initialize empty config map
        self.config_map = None

        # Create main parser
        self.parser = argparse.ArgumentParser(description="ttt arg parser.")

        if eval_mode:
            self._setup_eval_args()

        # Auto-generate argparse arguments from dataclass fields
        self._generate_args_from_dataclasses()

    def _setup_eval_args(self):
        """Add eval config to the config types."""
        self._config_types["eval"] = EvalConfig

    def _generate_args_from_dataclasses(self):
        """Generate argparse arguments automatically from dataclass fields and their metadata."""
        for config_name, config_class in self._config_types.items():
            # Get all fields in the dataclass
            fields = config_class.__dataclass_fields__  # type: ignore
            type_hints = get_type_hints(config_class)

            for field_name, field_info in fields.items():
                # Get the type and metadata
                field_type = type_hints.get(field_name)
                metadata = field_info.metadata
                default = field_info.default

                # Build argument name
                arg_name = f"--{config_name}.{field_name}"

                # Get help text and other metadata
                help_text = metadata.get("help", "")
                choices = metadata.get("choices", None)
                action = metadata.get("action", None)

                # Add the argument to the parser
                kwargs = {"help": help_text}

                # Handle different argument types
                if action:
                    # For boolean flags with store_true
                    kwargs["action"] = action
                    # Don't set default for store_true/store_false actions
                    if action not in ["store_true", "store_false"]:
                        kwargs["default"] = default
                else:
                    # For regular arguments
                    if field_name == "experimental.pipeline_parallel_split_points":
                        kwargs["type"] = string_list
                    elif field_type == Optional[str] and default is None:
                        kwargs["type"] = str
                        kwargs["default"] = None
                    elif field_type == Optional[int] and default is None:
                        kwargs["type"] = int
                        kwargs["default"] = None
                    else:
                        # Use the appropriate type
                        kwargs["type"] = type(default) if default is not None else field_type
                        kwargs["default"] = default

                    # Add choices if specified
                    if choices:
                        kwargs["choices"] = choices

                self.parser.add_argument(arg_name, **kwargs)

    def parse_args(self, args_list: list = sys.argv[1:]):
        args, cmd_args = self.parse_args_from_command_line(args_list)
        config_file = getattr(args, "job.config_file", None)
        # build up a two level dict
        args_dict = self._args_to_two_level_dict(args)
        if config_file is not None:
            try:
                with open(config_file, "rb") as f:
                    for k, v in tomllib.load(f).items():
                        # to prevent overwrite of non-specified keys
                        args_dict[k] |= v
            except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
                # logger.exception(f"Error while loading the configuration file: {config_file}")
                # logger.exception(f"Error details: {str(e)}")
                raise e

        # override args dict with cmd_args
        cmd_args_dict = self._args_to_two_level_dict(cmd_args)
        for section, section_args in cmd_args_dict.items():
            for k, v in section_args.items():
                args_dict[section][k] = v

        # Save for logging config later
        self.config_map = args_dict

        # Update the instance attributes with parsed values
        for config_name, config_values in args_dict.items():
            if config_name in self._config_types:
                config_class = self._config_types[config_name]
                # Check for unexpected keyword arguments
                valid_fields = set(config_class.__dataclass_fields__.keys())  # type: ignore
                provided_fields = set(config_values.keys())
                unexpected_fields = provided_fields - valid_fields

                if unexpected_fields:
                    raise TypeError(
                        f"Invalid field(s) found in {config_name} configuration: {', '.join(unexpected_fields)}.\n"
                        f"Valid fields for {config_name} are: {', '.join(sorted(valid_fields))}"
                    )

                # Create a new instance of the config dataclass with updated values
                config_instance = config_class(**config_values)
                setattr(self, config_name, config_instance)

        self._validate_config()

    def _args_to_two_level_dict(self, args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
        args_dict = defaultdict(dict)
        for k, v in vars(args).items():
            first_level_key, second_level_key = k.split(".", 1)
            args_dict[first_level_key][second_level_key] = v
        return args_dict

    def _validate_config(self):
        assert osp.isabs(self.job.dump_folder), "dump_folder should be an absolute path"

    def parse_args_from_command_line(self, args_list) -> Tuple[argparse.Namespace, argparse.Namespace]:
        """
        Parse command line arguments and return the parsed args and the command line only args
        """
        args = self.parser.parse_args(args_list)

        # aux parser to parse the command line only args, with no defaults from main parser
        aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        for arg, val in vars(args).items():
            if isinstance(val, bool):
                aux_parser.add_argument("--" + arg, action="store_true" if val else "store_false")
            elif arg == "experimental.pipeline_parallel_split_points":
                # without this special case, type inference breaks here,
                # since the inferred type is just 'list' and it ends up flattening
                # e.g. from ["layers.0", "layers.1"] into ["l", "a", "y", "e", "r", "s", ".0", ...]
                aux_parser.add_argument("--" + arg, type=string_list)
            else:
                aux_parser.add_argument("--" + arg, type=type(val))

        cmd_args, _ = aux_parser.parse_known_args(args_list)

        return args, cmd_args

    def to_dict(self) -> dict:
        """
        Returns the args of the instance as a nested dictionary.
        """
        assert self.config_map is not None, "Attempting to retrieve dict of args before parsing args."

        return self.config_map

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4)


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    print(config)
