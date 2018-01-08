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
f = tvm.build(s, [A, B], "opencl", target_host="llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a53 -mattr=+neon", name="myadd")
print(f.imported_modules[0].get_source())

# save files
temp = util.tempdir()
path_o = temp.relpath("myadd.o")
path_cl = temp.relpath("myadd.cl")
path_json = temp.relpath("myadd.tvm_meta.json")
f.save(path_o)
f.imported_modules[0].save(path_cl)

# replace host with the ip address of your device
host = '192.168.1.14'
port = 9090
# connect the remote device
remote = rpc.connect(host, port)

# upload files
remote.upload(path_o)
remote.upload(path_cl)
remote.upload(path_json)

# load files on remote device
fhost = remote.load_module("myadd.o")
fdev = remote.load_module("myadd.cl")
fhost.import_module(fdev)

# run
ctx = remote.cl(0)
a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
fhost(a, b)
#print(b.asnumpy())
#print(a.asnumpy())
#np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

time_f = fhost.time_evaluator(fhost.entry_name, ctx, number=10)
cost = time_f(a, b).mean
print('%g secs/op' % cost)
