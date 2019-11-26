import pyopencl as cl

mf = cl.mem_flags
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx) 
