from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn


class ForwardTrace(NamedTuple):
    module: str
    memory_diff: int
    memory_act: int
    module_params: int


class BackwardTrace(NamedTuple):
    module: str
    memory: int
    module_params: int


class MemoryMonitor:
    """
    Surround a module to get the graph of the memory consumption
    during the forward and backward, with a breakdown of the
    memory used of the activations versus the rest
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
        self.unit = 2 ** 20
        self._hooks = []
        self._prev_module_name = None
        self._memory_pre = 0
        self._traced_module_names = set()

    def prepare(self):
        torch.cuda.empty_cache()

    def monitor(self, model: nn.Module):
        for name, m in model.named_modules():
            h1 = m.register_forward_pre_hook(self._create_pre_forward_hook(name))
            h2 = m.register_forward_hook(self._create_post_forward_hook(name))
            h3 = m.register_backward_hook(self._create_backward_hook(name))
            self._hooks.extend([h1, h2, h3])

    def reset(self):
        for h in self._hooks:
            h.remove()

    def _create_pre_forward_hook(self, name: str):
        def _pre_forward_hook(module: nn.Module, inputs):
            allocated_mb, reserved_mb = self._capture_memory(module, trace=False)
            self._prev_module_name = name
            self._memory_pre = allocated_mb
        return _pre_forward_hook

    def _create_post_forward_hook(self, name: str):
        def _post_forward_hook(module: nn.Module, inputs, outputs):

            if name == self._prev_module_name:
                allocated_mb, reserved_mb = self._capture_memory(module)
                self._traced_module_names.add(name)

                # Get the memory allocated for output activations
                ys = self._filter_allocated_output(inputs, outputs)
                memory_act = sum(self._get_module_output_size(y) for y in ys)

                # Compute the memory diff + memory taken by the activations
                self.activations.append(ForwardTrace(
                    module=str(module),
                    memory_diff=allocated_mb - self._memory_pre,
                    memory_act=memory_act // self.unit,
                    module_params=self.get_parameter_size(module) // self.unit,
                ))
                self.gradients.append(BackwardTrace(
                    module=str(module),
                    memory=0,
                    module_params=0,
                ))
            self._prev_module_name = None
            self._memory_pre = 0

        return _post_forward_hook

    def _create_backward_hook(self, name: str):
        def _backward_hook(module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor):
            if name not in self._traced_module_names:
                return

            ys = self._filter_allocated_output(grad_input, grad_output)
            memory = sum(self._get_module_output_size(y) for y in ys)

            allocated_mb, reserved_mb = self._capture_memory(module)
            self.activations.append(ForwardTrace(
                module=str(module),
                memory_diff=0,
                memory_act=0,
                module_params=0,
            ))
            self.gradients.append(BackwardTrace(
                module=str(module),
                memory=memory // self.unit,
                module_params=self.get_parameter_size(module) // self.unit,
            ))
        return _backward_hook

    def _capture_memory(self, module: nn.Module, trace: bool = True):
        torch.cuda.synchronize()
        allocated_mb = torch.cuda.memory_allocated() // self.unit
        reserved_mb = torch.cuda.memory_reserved() // self.unit
        if trace:
            self.memory_traces.append({
                'mem_alloc': allocated_mb,
                'mem_cached': reserved_mb,
                'module': str(module)
            })
        return allocated_mb, reserved_mb

    @property
    def total_forward_diff(self):
        return sum(a.memory_diff for a in self.activations)

    @property
    def total_activation(self):
        return sum(a.memory_act for a in self.activations)

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
            'total_forward_diff': self.total_forward_diff,
        }

    def top_activation_producers(self, top=10):
        return sorted(self.activations, key=lambda a: a.memory, reverse=True)[:top]

    def show_plots(self, figsize=(12, 12), cumul: bool = False, capture: bool = False):
        import matplotlib.pyplot as plt
        ncols, nrows = 2, 3

        fig, ax = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows)
        ax[0, 0].set_title('memory allocated')
        ax[0, 0].plot([trace['mem_alloc'] for trace in self.memory_traces])
        if cumul:
            cum_memory_diff = np.cumsum([a.memory_diff for a in self.activations])
            cum_memory_act = np.cumsum([a.memory_act for a in self.activations])
            cum_sum_grad_params = np.cumsum([g.module_params for g in self.gradients])
            ax[0, 0].plot(cum_memory_diff)
            ax[0, 0].plot(cum_memory_act)
            ax[0, 0].plot(cum_sum_grad_params)
        ax[0, 1].set_title('memory cached')
        ax[0, 1].plot([trace['mem_cached'] for trace in self.memory_traces])
        ax[1, 0].set_title('activation allocations')
        ax[1, 0].plot([a.memory_diff for a in self.activations])
        ax[1, 0].plot([a.memory_act for a in self.activations])
        ax[1, 1].set_title('parameter memory')
        ax[1, 1].plot([a.module_params for a in self.activations])
        ax[2, 0].set_title('gradient allocations')
        ax[2, 0].plot([g.memory for g in self.gradients])
        ax[2, 1].set_title('gradient params')
        ax[2, 1].plot([g.module_params for g in self.gradients])
        if not capture:
            plt.show()
        else:
            return matplotlib_figure_to_image(fig)

    @staticmethod
    def get_parameter_count(module):
        return sum(p.numel() for p in module.parameters())

    @classmethod
    def get_parameter_size(cls, module):
        return sum(p.numel() * cls._get_dtype_size(p) for p in module.parameters())

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
    def _get_module_output_size(cls, xs):
        if isinstance(xs, torch.Tensor):
            x = xs
            p = cls._get_dtype_size(x)
            for x in x.shape:
                p *= x
            return p
        elif isinstance(xs, tuple) or isinstance(xs, list):
            return sum(cls._get_module_output_size(x) for x in xs)
        return 0

    @classmethod
    def _get_dtype_size(cls, x: torch.Tensor):
        return 2 if x.dtype == torch.float16 else 4

    @staticmethod
    def _is_same_storage(x: torch.Tensor, y: torch.Tensor):
        return x.storage().data_ptr() == y.storage().data_ptr()

    @staticmethod
    def _collect_tensors(module_outputs):
        tensors = []
        to_visit = [module_outputs]
        while to_visit:
            x = to_visit.pop()
            if isinstance(x, torch.Tensor):
                tensors.append(x)
            elif isinstance(x, tuple) or isinstance(x, list):
                to_visit.extend(module_outputs)
        return tensors

    @classmethod
    def _filter_allocated_output(cls, inputs, outputs):
        xs = cls._collect_tensors(inputs)
        ys = cls._collect_tensors(outputs)
        return [y for y in ys if all(not cls._is_same_storage(x, y) for x in xs)]


def matplotlib_figure_to_image(fig):
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
