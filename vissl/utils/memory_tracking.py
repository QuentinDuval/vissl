from typing import NamedTuple, List, Union

import numpy as np
import torch
import torch.nn as nn


class ActivationTrace(NamedTuple):
    module: str
    memory: int
    params: int
    inputs: Union[List[torch.Tensor], torch.Tensor]
    outputs: Union[List[torch.Tensor], torch.Tensor]


class GradientTrace(NamedTuple):
    module: str
    memory: int
    params: int
    inputs: Union[List[torch.Tensor], torch.Tensor]
    outputs: Union[List[torch.Tensor], torch.Tensor]


class MemoryProfiler:
    """
    Surround a module to get the graph of memory consumption of the activations
    """

    # TODO - differentiate the allocation from the memory allocation

    # TODO - feedback:
    #  * seems like the curves at each layer are important
    #  * getting an idea of what is being allocated during forward: activations (just take the output)
    #  * getting an idea of what is being allocated during backward: gradient activations

    # TODO - interesting options:
    #  * warm-up period? the first step is weird but also interesting
    #  * measuring the amount of memory used for optimizer / grad (which is supposed to be constant)

    # TODO - deal with the backward for the activations as well (to get a better view of what happens with checkpointing)

    def __init__(self):
        self.activations = []
        self.gradients = []
        self.memory_traces = []

        self._hooks = []
        self._prev_module_type = None
        self._memory_pre = 0
        self._traced_module_types = set()

    def prepare(self):
        torch.cuda.empty_cache()

    def monitor(self, model: nn.Module):
        for name, m in model.named_modules():
            h1 = m.register_forward_pre_hook(self._pre_forward_hook)
            h2 = m.register_forward_hook(self._post_forward_hook)
            h3 = m.register_backward_hook(self._backward_hook)
            self._hooks.extend([h1, h2, h3])

    def reset(self):
        for h in self._hooks:
            h.remove()

    def _pre_forward_hook(self, module: nn.Module, inputs):
        """
        Called before the forward of a module
        """
        allocated_mb, reserved_mb = self._capture_memory(module, trace=False)
        self._prev_module_type = type(module)
        self._memory_pre = allocated_mb

    def _post_forward_hook(self, module: nn.Module, inputs, outputs):
        """
        Called after the forward of a module
        """
        if type(module) == self._prev_module_type:
            allocated_mb, reserved_mb = self._capture_memory(module)
            self._traced_module_types.add(type(module))
            self.activations.append(ActivationTrace(
                module=str(module),
                memory=allocated_mb - self._memory_pre,
                params=self.get_parameter_size(module) // 2 ** 20,
                inputs=self._get_forward_shapes(inputs),
                outputs=self._get_forward_shapes(outputs),
            ))
            self.gradients.append(GradientTrace(
                module=str(module),
                memory=0,
                params=0,
                inputs=[],
                outputs=[],
            ))
        self._prev_module_type = None
        self._memory_pre = 0

    def _backward_hook(self, module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor):
        """
        Called after the computation of the input gradients by the module
        """
        if type(module) in self._traced_module_types:
            allocated_mb, reserved_mb = self._capture_memory(module)
            self.activations.append(ActivationTrace(
                module=str(module),
                memory=0,
                params=0,  # sum(p.numel() for p in module.parameters()),
                inputs=[],
                outputs=[],
            ))
            self.gradients.append(GradientTrace(
                module=str(module),
                memory=self._get_gradient_size(grad_input) // 2 ** 20,
                # What is allocated (but need to find what is kept - since only param grads are kept?)
                params=self.get_parameter_size(module) // 2 ** 20,
                inputs=self._get_gradient_shapes(grad_input),
                outputs=self._get_gradient_shapes(grad_output),
            ))

    def _capture_memory(self, module: nn.Module, trace: bool = True):
        torch.cuda.synchronize()
        allocated_mb = torch.cuda.memory_allocated() // 2 ** 20
        reserved_mb = torch.cuda.memory_reserved() // 2 ** 20
        if trace:
            self.memory_traces.append({
                'mem_alloc': allocated_mb,
                'mem_cached': reserved_mb,
                'module': str(module)
            })
        return allocated_mb, reserved_mb

    @property
    def total_activation(self):
        return sum(a.memory for a in self.activations)

    @property
    def max_memory_allocated(self):
        return max(t['mem_alloc'] for t in self.memory_traces)

    @property
    def max_memory_cached(self):
        return max(t['mem_cached'] for t in self.memory_traces)

    @property
    def summary(self):
        return {
            'max_memory_allocated': self.max_memory_allocated,
            'max_memory_cached': self.max_memory_cached,
            'total_activation': self.total_activation,
        }

    def top_activation_producers(self, top=10):
        return sorted(self.activations, key=lambda a: a.memory, reverse=True)[:top]

    def show_plots(self, figsize=(12, 12), cumul: bool = False):
        import matplotlib.pyplot as plt
        ncols, nrows = 2, 3

        fig, ax = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows)
        ax[0, 0].set_title('memory allocated')
        ax[0, 0].plot([trace['mem_alloc'] for trace in self.memory_traces])
        if cumul:
            cum_sum_activations = np.cumsum([a.memory for a in self.activations])
            cum_sum_grad_params = np.cumsum([g.params for g in self.gradients])
            ax[0, 0].plot(cum_sum_activations)
            ax[0, 0].plot(cum_sum_grad_params)
        ax[0, 1].set_title('memory cached')
        ax[0, 1].plot([trace['mem_cached'] for trace in self.memory_traces])
        ax[1, 0].set_title('activation allocations')
        ax[1, 0].plot([a.memory for a in self.activations])
        ax[1, 1].set_title('parameter memory')
        ax[1, 1].plot([a.params for a in self.activations])
        ax[2, 0].set_title('gradient allocations')
        ax[2, 0].plot([g.memory for g in self.gradients])
        ax[2, 1].set_title('gradient params')
        ax[2, 1].plot([g.params for g in self.gradients])
        plt.show()

    @staticmethod
    def get_parameter_count(module):
        return sum(p.numel() for p in module.parameters())

    @staticmethod
    def get_parameter_size(module):
        return sum(p.numel() * (2 if p.dtype == torch.float16 else 4) for p in module.parameters())

    @classmethod
    def _get_forward_shapes(cls, xs):
        if isinstance(xs, torch.Tensor):
            return xs.shape
        else:
            return [cls._get_forward_shapes(x) for x in xs]

    @classmethod
    def _get_gradient_shapes(cls, xs):
        if isinstance(xs, torch.Tensor):
            return xs.shape
        elif isinstance(xs, tuple) or isinstance(xs, list):
            return [cls._get_gradient_shapes(x) for x in xs if x is not None]
        return None

    @classmethod
    def _get_gradient_size(cls, xs):
        if isinstance(xs, torch.Tensor):
            x = xs
            p = 2 if x.dtype == torch.float16 else 4
            for x in x.shape:
                p *= x
            return p
        elif isinstance(xs, tuple) or isinstance(xs, list):
            return sum(cls._get_gradient_size(x) for x in xs)
        return 0
