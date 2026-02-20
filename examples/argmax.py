import pyopencl as cl
import pyopencl.array
import pyopencl.reduction
import numpy as np
import os
os.environ['PYOPENCL_COMPILER_OUTPUT']='0'

def argmax(vec_in, approx):
    #vec_np = vec_in.copy()
    vec_h = vec_in*(10**approx)
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    s_h = np.intc([vec_in.size], dtype = np.intc)
    s_g = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = s_h)
    vec = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = vec_h.astype(np.float32))
    s  = cl.array.Array(queue, s_h.shape, dtype = np.intc, data = s_g)
    vec = cl.array.Array(queue, vec_in.shape, dtype = np.float32, data = vec)
    kernel = cl.reduction.ReductionKernel(context, np.float32, neutral='-1/0',
                                          map_expr='((int)(x[i]))*(*si) + i',
                                          reduce_expr = 'isgreater(a, b)*a + (1-isgreater(a,b))*b',
                                          arguments = '__global float *x, __global int *si')
    
    return int(kernel(vec, s).get() % vec_in.size)

t = np.random.random(5)
print(t)
print('got: {}'.format(argmax(t, 4)))
