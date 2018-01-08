from __future__ import absolute_import, print_function

import tvm
import numpy as np
from tvm.contrib import rpc, util

n = tvm.convert(1024)
A = tvm.placeholder((n,), name='A')
B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
s = tvm.create_schedule(B.op)

# build kernel (different from cpu, we need bind axis for OpenCL)
xo, xi = s[B].split(B.op.axis[0], factor=32)
s[B].bind(xo, tvm.thread_axis("blockIdx.x"))
s[B].bind(xi, tvm.thread_axis("threadIdx.x"))
f = tvm.build(s, [A, B], "opencl", target_host="llvm", name="myadd")

# run
ctx = tvm.cl(0)
a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
f(a, b)
np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

print(f.imported_modules[0].get_source())

time_f = f.time_evaluator(f.entry_name, ctx, number=10)
cost = time_f(a, b).mean
print('%g secs/op' % cost)
