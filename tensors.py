import numpy as np
import pyopencl as cl

from context import mf, ctx, queue 
from kernels import build_kernel

def tensor(ar: np.ndarray, dtype=np.float64):
    ar = ar.astype(dtype)
    ar_gpu = ar.reshape(np.prod(ar.shape))
    buff = \
        cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=ar_gpu,
        )
    return Tensor(ar.shape, ar_gpu.shape, buff, dtype=dtype)

class Tensor:
    def __init__(self, shape: tuple, shape_gpu: tuple,
                 buff: cl._cl.Buffer, dtype: type):
        self.shape = shape
        self.dtype = dtype
        self._shape_gpu = shape_gpu
        self._tensor_gpu = buff
        self._kernel = build_kernel(dtype)
        
    def __str__(self):
        return f'{self.dtype} tensor of shape {self.shape}'
    
    def to_cpu(self):
        res = np.empty(shape=self.shape).astype(self.dtype)
        cl.enqueue_copy(queue, res, self._tensor_gpu)
        return res
    
    def __add__(self, tens2):
        if self.dtype != tens2.dtype:
            raise TypeError(
                f'First tensor is {self.dtype}, ' \
                f'second tensor is {tens2.dtype}'
            )
        r = np.empty(shape=self.shape).astype(self.dtype)
        res_g = cl.Buffer(ctx, mf.WRITE_ONLY, r.nbytes)
        self._kernel.sum(
            queue, self._shape_gpu, None,
            self._tensor_gpu,
            tens2._tensor_gpu,
            res_g,
        )
        return Tensor(self.shape, self._shape_gpu, res_g, self.dtype)
    
    def __mul__(self, num: int):
        num = np.int64(num)
        r = np.empty(shape=self.shape).astype(self.dtype)
        res_g = cl.Buffer(ctx, mf.WRITE_ONLY, r.nbytes)
        self._kernel.mult(
            queue, self._shape_gpu, None,
            self._tensor_gpu, num, res_g,
        )
        return Tensor(self.shape, self._shape_gpu, res_g, self.dtype)
    
    def tensor_product(self, tens2):
        if self.dtype != tens2.dtype:
            raise TypeError(
                f'First tensor is {self.dtype}, ' \
                f'second tensor is {tens2.dtype}'
            )
        r = np.zeros(
            shape=self.shape + tens2.shape
        ).astype(self.dtype)
        #shape_gpu = (np.prod(self._shape_gpu + tens2._shape_gpu),)
        #print(self._shape_gpu)
        #print(tens2._shape_gpu)
        res_g = cl.Buffer(ctx, mf.WRITE_ONLY, r.nbytes)
        self._kernel.tensor_product(
            queue, r.shape,
            None, np.int32(tens2._shape_gpu[0]),
            self._tensor_gpu, tens2._tensor_gpu, res_g,
        )
        return Tensor(r.shape, (np.prod(r.shape),), res_g, self.dtype)
