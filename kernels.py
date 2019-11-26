import pyopencl as cl
import numpy as np

from context import ctx

types_mapping = {
    np.float32: 'float',
    np.float64: 'double',
    np.int32: 'int',
    np.int64: 'long',
}

kernel = \
'''
__kernel void tensor_product(
    const unsigned int size,
    __global const {dtype} *m1,
    __global const {dtype} *m2,
    __global {dtype} *res){{
        int i = get_global_id(1); 
        int j = get_global_id(0);
        res[i + size * j] = m1[j] * m2[i];
    }}

__kernel void sum(
    __global const {dtype} *m1,
    __global const {dtype} *m2,
    __global {dtype} *res){{
        int gid = get_global_id(0);
        res[gid] = m1[gid] + m2[gid];
    }}
'''

def build_kernel(dtype):
    return cl.Program(ctx, kernel.format(dtype=types_mapping[dtype])).build()
