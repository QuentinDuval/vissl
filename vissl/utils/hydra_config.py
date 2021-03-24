# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import pprint
import sys
from typing import Any, List

from omegaconf import DictConfig, OmegaConf
from vissl.config import AttrDict, auto_complete_configuration, check_cfg_version


class HydraConfig:
    """
    Utility class to load / compose VISSL configurations, dealing with
    all the underlying details such as:

    - merging with the default configuration
    - conversion of the configuration to `AttrDict`
    - upgrading the configuration to newest version
    - inference of parameter values
    """

    @staticmethod
    def from_file(config_path: str):
        return HydraConfig.from_command_line(overrides=[f"config={config_path}"])

    @staticmethod
    def from_command_line(overrides: List[Any]):
        from hydra.experimental import compose, initialize_config_module

        with initialize_config_module(config_module="vissl.config"):
            cfg = compose("defaults", overrides=overrides)
        args, config = convert_to_attrdict(cfg)
        return args, config


def convert_to_attrdict(cfg: DictConfig, cmdline_args: List[Any] = None):
    """
    Given the user input Hydra Config, and some command line input options
    to override the config file:
    1. merge and override the command line options in the config
    2. Convert the Hydra OmegaConf to AttrDict structure to make it easy
       to access the keys in the config file
    3. Also check the config version used is compatible and supported in vissl.
       In future, we would want to support upgrading the old config versions if
       we make changes to the VISSL default config structure (deleting, renaming keys)
    4. We infer values of some parameters in the config file using the other
       parameter values.
    """
    if cmdline_args:
        # convert the command line args to DictConfig
        sys.argv = cmdline_args
        cli_conf = OmegaConf.from_cli(cmdline_args)

        # merge the command line args with config
        cfg = OmegaConf.merge(cfg, cli_conf)

    # convert the config to AttrDict
    cfg = OmegaConf.to_container(cfg)
    cfg = AttrDict(cfg)

    # check the cfg has valid version
    check_cfg_version(cfg)

    # assert the config and infer
    config = cfg.config
    auto_complete_configuration(config)
    download_weight_parameters(config)
    return cfg, config


def is_hydra_available():
    """
    Check if Hydra is available. Simply python import to test.
    """
    try:
        import hydra  # NOQA

        hydra_available = True
    except ImportError:
        hydra_available = False
    return hydra_available


def print_cfg(cfg):
    """
    Supports printing both Hydra DictConfig and also the AttrDict config
    """
    logging.info("Training with config:")
    if isinstance(cfg, DictConfig):
        logging.info(cfg.pretty())
    else:
        logging.info(pprint.pformat(cfg))


def download_weight_parameters(cfg):
    """
    If the user has specified the model initialization from a params_file, we check if
    the params_file is a url. If it is, we download the file to a local cache directory
    and use that instead
    """
    from vissl.utils.checkpoint import get_checkpoint_folder
    from vissl.utils.io import cache_url, is_url

    if is_url(cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE):
        checkpoint_dir = get_checkpoint_folder(cfg)
        cache_dir = f"{checkpoint_dir}/params_file_cache/"
        cached_url_path = cache_url(
            url=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE, cache_dir=cache_dir
        )
        cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE = cached_url_path
