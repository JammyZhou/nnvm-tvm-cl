Overview
============
This repository is intended for effort on enabling OpenCL support of NNVM/TVM on ARM platforms. Currently Bubblegum96 96Boards with PowerVR Rogue G6230 GPU is used.

TOPI Unit Tests
===================
https://github.com/dmlc/tvm/tree/master/topi/tests/python
- broadcast [Pass]
- conv2d_hwcn [https://github.com/JammyZhou/nnvm-tvm-cl/issues/1]
- conv2d_nchw [https://github.com/JammyZhou/nnvm-tvm-cl/issues/2]
- conv2d_transpose_nchw [similar as conv2d_hwcn]
- dense [Pass]
- depthwise_conv2d_back_input [https://github.com/JammyZhou/nnvm-tvm-cl/issues/3]
- depthwise_conv2d_back_weight [similar as conv2d_hwcn]
- depthwise_conv2d [similar as conv2d_nchw]
- pooling [Pass]
- reduce [https://github.com/JammyZhou/nnvm-tvm-cl/issues/4]
- relu [Pass]
- softmax [similar as reduce]
- transform [https://github.com/JammyZhou/nnvm-tvm-cl/issues/5]

End-To-End
============
- MXNet resnet-18 CL_OUT_OF_RESOURCES [https://github.com/dmlc/tvm/issues/761]

References
============
- http://docs.tvmlang.org/
- http://nnvm.tvmlang.org/index.html
- https://mxnet.incubator.apache.org/install/index.html
