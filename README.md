Overview
============
This repository is intended for effort on enabling OpenCL support of NNVM/TVM on ARM platforms. Currently Bubblegum96 96Boards is used.

TOPI Unit Tests
===================
https://github.com/dmlc/tvm/tree/master/topi/tests/python
- broadcast
- conv2d_hwcn
- conv2d_nchw
- conv2d_transpose_nchw
- dense
- depthwise_conv2d_back_input
- depthwise_conv2d_back_weight
- depthwise_conv2d
- pooling
- reduce
- relu
- softmax
- transform

Known issues
============
- CL_OUT_OF_RESOURCES when deploy resnet - https://github.com/dmlc/tvm/issues/761

References
============
- http://docs.tvmlang.org/
- http://nnvm.tvmlang.org/index.html
- https://mxnet.incubator.apache.org/install/index.html
