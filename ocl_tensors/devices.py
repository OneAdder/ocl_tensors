import os
import pyopencl as cl
from typing import List


def list_devices() -> List[cl.Platform]:
    return cl.get_platforms()


class Device:
    _types_mapping = {
        'float32': 'float',
        'float64': 'double',
        'int32': 'int',
        'int64': 'long',
    }
    _busy = {}

    def __init__(self, device: cl.Platform):
        if device in self._busy:
            self.ctx, self.kernels = self._busy[device]
        else:
            self.ctx = \
                cl.Context(
                    dev_type=cl.device_type.ALL,
                    properties=[(cl.context_properties.PLATFORM, device)],
                )
            self.kernels = self._build_kernels()
            self._busy.update({device: (self.ctx, self.kernels)})
        self.queue = cl.CommandQueue(self.ctx)
        self.device = device

    def __repr__(self):
        return str(self.device)

    def _build_kernels(self):
        with open(os.path.join(os.path.dirname(__file__), 'kernel.cl')) as f:
            kernel = f.read()
        built_kernels = {
            t_np: cl.Program(
                self.ctx, kernel.replace('real_number', t_cl)).build()
            for t_np, t_cl in self._types_mapping.items()
        }
        return built_kernels
