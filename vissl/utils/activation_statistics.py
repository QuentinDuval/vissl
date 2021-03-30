# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import abc
from typing import NamedTuple, Set

import numpy as np
import torch
import torch.nn as nn


class ActivationStatistics(NamedTuple):
    """
    Information collected on each activation:
    - "name" of the module and "module_type"
    - current "iteration" of training
    - mean, min and max statistics
    """

    name: str
    iteration: int
    module_type: str
    mean: float
    maxi: float
    mini: float


class ActivationStatisticsObserver(abc.ABC):
    """
    Abstract interface to override to either collect or stream
    statistics as they are produced
    """

    @abc.abstractmethod
    def consume(self, stat: ActivationStatistics):
        pass


class ActivationStatisticsMonitor:
    """
    Watch via hooks the content of the model's activations during training
    computes basic statistics on them, and stream them to an 'observer'.

    This implementation only traces modules which:
    - do not have child modules (ex: nn.Sequential modules are ignored)
    - are in training mode (the goal is to identify divergences)

    Depending on the 'observer' implementation, the results can be
    accumulated or streamed to tensorboard (or any visualisation tool).
    """

    def __init__(
        self,
        observer: ActivationStatisticsObserver,
        log_frequency: int,
        ignored_modules: Set[str] = None,
    ):
        self.observer = observer
        self.log_frequency = log_frequency
        self.ignored_modules = ignored_modules
        if self.ignored_modules is None:
            self.ignored_modules = {"torch.nn.modules.activation.ReLU"}
        self.iteration = -1
        self._hooks = []
        self._prev_module_name = None

    def set_iteration(self, iteration: int):
        self.iteration = iteration

    def monitor(self, model: nn.Module):
        for name, m in model.named_modules():
            if self._get_qualified_type(m) not in self.ignored_modules:
                h1 = m.register_forward_pre_hook(self._create_pre_forward_hook(name))
                h2 = m.register_forward_hook(self._create_post_forward_hook(name))
                self._hooks.extend([h1, h2])

    def reset(self):
        for h in self._hooks:
            h.remove()
        self.iteration = -1
        self._prev_module_name = None

    def _should_log(self, module: nn.Module):
        return self.iteration % self.log_frequency == 0 and module.training

    def _create_pre_forward_hook(self, name: str):
        def _pre_forward_hook(module: nn.Module, inputs):
            if self._should_log(module):
                self._prev_module_name = name
            else:
                self._prev_module_name = None

        return _pre_forward_hook

    def _create_post_forward_hook(self, name: str):
        def _post_forward_hook(module: nn.Module, inputs, outputs):
            # Eliminate non-leaf modules as well as modules ignored by the forward
            if self._prev_module_name != name:
                return

            outputs = self._collect_tensors(outputs, detach=True)
            mean = np.mean([o.mean().item() for o in outputs])
            maxi = np.max([o.max().item() for o in outputs])
            mini = np.min([o.min().item() for o in outputs])
            self.observer.consume(
                ActivationStatistics(
                    name=name,
                    iteration=self.iteration,
                    module_type=self._get_qualified_type(module),
                    mean=mean,
                    maxi=maxi,
                    mini=mini,
                )
            )

        return _post_forward_hook

    @staticmethod
    def _get_qualified_type(module: nn.Module):
        return type(module).__module__ + "." + type(module).__name__

    @staticmethod
    def _collect_tensors(xs, detach=False):
        tensors = []
        to_visit = [xs]
        while to_visit:
            x = to_visit.pop()
            if isinstance(x, torch.Tensor):
                if detach:
                    x = x.detach()
                tensors.append(x)
            elif isinstance(x, tuple) or isinstance(x, list):
                to_visit.extend(xs)
        return tensors


class ActivationStatisticsAccumulator(ActivationStatisticsObserver):
    """
    Implementation of ActivationStatisticsObserver which collects
    the statistics in a list (especially useful for tests)
    """

    def __init__(self):
        self.stats = []

    def consume(self, stat: ActivationStatistics):
        self.stats.append(stat)