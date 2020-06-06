import numpy as np
import pyopencl as cl
from typing import Iterable, Union

from ocl_tensors.devices import Device

mf = cl.mem_flags


def tensor(data: Iterable,
           dtype: type = None,
           device: cl.Platform = None):
    if isinstance(data, np.ndarray):
        np_array = data.astype(dtype) if dtype else data
    else:
        np_array = np.array(data, dtype=dtype) if dtype else np.array(data)
    np_array_gpu = np_array.reshape(np.prod(np_array.shape))
    if not device:
        device = cl.get_platforms()[0]
    t_device = Device(device)
    buff = \
        cl.Buffer(
            t_device.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=np_array_gpu,
        )
    new_tensor = Tensor(np_array.shape, np_array_gpu.shape,
                        buff, t_device, dtype=np_array.dtype)
    return new_tensor


class Tensor:
    def __init__(self, shape: tuple, shape_gpu: tuple,
                 buff: cl._cl.Buffer, device: Device,
                 dtype: Union[np.dtype, type]):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._shape_gpu = shape_gpu
        self._tensor_gpu = buff
        self._kernel = device.kernels[str(dtype)]
        
    def __str__(self):
        return f'{self.dtype} tensor of shape {self.shape}'
    
    def to_cpu(self) -> np.ndarray:
        res = np.empty(shape=self.shape).astype(self.dtype)
        cl.enqueue_copy(self.device.queue, res, self._tensor_gpu)
        return res

    def _check_same_type(self, other: "Tensor"):
        if self.dtype != other.dtype:
            raise TypeError(
                f'First tensor is {self.dtype}, '
                f'second tensor is {other.dtype}'
            )

    def __add__(self, other: Union[Union[int, float], "Tensor"]) -> "Tensor":
        if isinstance(other, int) or isinstance(other, float):
            new_tensor = self.scalar_sum(other)
        elif self.shape == other.shape:
            new_tensor = self.tensor_sum(other)
        else:
            raise ValueError('Unable to find an available operation')
        return new_tensor

    def __mul__(self, other: Union[Union[int, float], "Tensor"]) -> "Tensor":
        if isinstance(other, int) or isinstance(other, float):
            return self.scalar_mult(other)
        self._check_same_type(other)
        return self.inner_product(other)

    def __pow__(self, other):
        self._check_same_type(other)
        if len(self.shape) == 1 and len(other.shape) == 1:
            new_tensor = self.outer_product(other)
        elif len(self.shape) == 2 and len(other.shape) == 2:
            new_tensor = self.matmul(other)
        else:
            raise ValueError(
                'Unable to calculate product of matrices of shapes'
                '{} and {}'.format(self.shape, other.shape)
            )
        return new_tensor

    def scalar_sum(self, num: Union[int, float]) -> "Tensor":
        num = self.dtype.type(num)
        r = np.empty(shape=self.shape).astype(self.dtype)
        res_g = cl.Buffer(self.device.ctx, mf.WRITE_ONLY, r.nbytes)
        self._kernel.scalar_sum(
            self.device.queue, self._shape_gpu, None,
            self._tensor_gpu, num, res_g,
        )
        t = Tensor(self.shape, self._shape_gpu, res_g, self.device, self.dtype)
        return t
    
    def scalar_mult(self, num: Union[int, float]) -> "Tensor":
        num = self.dtype.type(num)
        r = np.empty(shape=self.shape).astype(self.dtype)
        res_g = cl.Buffer(self.device.ctx, mf.WRITE_ONLY, r.nbytes)
        self._kernel.scalar_mult(
            self.device.queue, self._shape_gpu, None,
            self._tensor_gpu, num, res_g,
        )
        t = Tensor(self.shape, self._shape_gpu, res_g, self.device, self.dtype)
        return t

    def tensor_sum(self, other: "Tensor") -> "Tensor":
        self._check_same_type(other)
        r = np.empty(shape=self.shape).astype(self.dtype)
        res_g = cl.Buffer(self.device.ctx, mf.WRITE_ONLY, r.nbytes)
        self._kernel.tensor_sum(
            self.device.queue, self._shape_gpu, None,
            self._tensor_gpu, other._tensor_gpu, res_g,
        )
        t = Tensor(self.shape, self._shape_gpu, res_g, self.device, self.dtype)
        return t

    def inner_product(self, other: "Tensor") -> "Tensor":
        self._check_same_type(other)
        r = np.empty(shape=self.shape).astype(self.dtype)
        res_g = cl.Buffer(self.device.ctx, mf.WRITE_ONLY, r.nbytes)
        self._kernel.inner_product(
            self.device.queue, self._shape_gpu, None,
            self._tensor_gpu,
            other._tensor_gpu,
            res_g,
        )
        return Tensor(self.shape, self._shape_gpu, res_g, self.device, self.dtype)

    def matmul(self, other: "Tensor") -> "Tensor":
        self._check_same_type(other)
        if not self.shape[-1] == other.shape[0]:
            raise ValueError(
                'Unable to multiply matrices of shapes {} and {}'.format(
                    self.shape, other.shape)
            )
        r = np.empty(
            shape=(self.shape[0], other.shape[-1])
        ).astype(self.dtype)
        dest_buf = cl.Buffer(self.device.ctx, mf.WRITE_ONLY, r.nbytes)
        self._kernel.matmul(
            self.device.queue, (other.shape[-1], self.shape[0]), None,
            np.int32(self.shape[-1]), np.int32(other.shape[-1]),
            np.int32(other.shape[-1]), self._tensor_gpu,
            other._tensor_gpu, dest_buf,
        )
        t = Tensor(
            r.shape, (np.prod(r.shape),),
            dest_buf, self.device, self.dtype,
        )
        return t

    def outer_product(self, other: "Tensor") -> "Tensor":
        self._check_same_type(other)
        if len(self.shape) > 1 or len(other.shape) > 1:
            raise NotImplementedError('Tensor product is not implemented yet')
        r = np.empty(shape=(self.shape[0], other.shape[0])).astype(self.dtype)
        dest_buf = cl.Buffer(self.device.ctx, mf.WRITE_ONLY, r.nbytes)
        self._kernel.matmul(
            self.device.queue, (other.shape[-1], self.shape[0]), None,
            np.int32(1), np.int32(other.shape[-1]),
            np.int32(other.shape[-1]), self._tensor_gpu,
            other._tensor_gpu, dest_buf,
        )
        return Tensor(r.shape, (np.prod(r.shape),), dest_buf, self.device, self.dtype)
