import mxnet as mx
import nnvm
import tvm
import numpy as np

# Use "Agg" backend of matplotlib to make it work with ssh
import matplotlib
matplotlib.use('Agg')

from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.utils import download
from PIL import Image
from matplotlib import pyplot as plt

block = get_model('resnet18_v1', pretrained=True)
img_name = 'cat.jpg'
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'synset.txt'
download('https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true', img_name)
download(synset_url, synset_name)
with open(synset_name) as f:
    synset = eval(f.read())
image = Image.open(img_name).resize((224, 224))
plt.imshow(image)
plt.show()

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

x = transform_image(image)
print('x', x.shape)

sym, params = nnvm.frontend.from_mxnet(block)
# we want a probability so add a softmax operator
sym = nnvm.sym.softmax(sym)

import nnvm.compiler
target = 'opencl'
shape_dict = {'data': x.shape}
graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

from tvm.contrib import graph_runtime
ctx = tvm.cl(0)
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)
# set inputs
m.set_input('data', tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
# execute
m.run()
# get outputs
tvm_output = m.get_output(0, tvm.nd.empty((1000,), dtype))
top1 = np.argmax(tvm_output.asnumpy())
print('TVM prediction top-1:', top1, synset[top1])
