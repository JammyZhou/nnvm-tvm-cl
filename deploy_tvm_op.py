from __future__ import absolute_import, print_function

import tvm
import numpy as np
from tvm.contrib import rpc, util

n = tvm.convert(1024)
A = tvm.placeholder((n,), name='A')
B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
s = tvm.create_schedule(B.op)

f = tvm.build(s, [A, B], target='llvm', name='myadd')

ctx = tvm.cpu(0)
a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
f(a, b)
np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

time_f = f.time_evaluator(f.entry_name, ctx, number=10)
cost = time_f(a, b).mean
print('%g secs/op' % cost)
