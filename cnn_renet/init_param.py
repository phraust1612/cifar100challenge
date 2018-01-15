import numpy as np
import caffe
import os

def extract_caffe_model(model, weights, output_path):
  """extract caffe model's parameters to numpy array, and write them to files
  Args:
    model: path of '.prototxt'
    weights: path of '.caffemodel'
    output_path: output path of numpy params 
  Returns:
    None
  """
  net = caffe.Net(model, caffe.TEST)
  net.copy_from(weights)

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  for item in net.params.items():
    name, layer = item
    print('convert layer: ' + name)

    num = 0
    for p in net.params[name]:
      np.save(output_path + '/' + str(name) + '_' + str(num), p.data)
      num += 1

def init_resnet (output):
  d = 'cnn_resnet/param_resnet/'
  l = os.listdir(d)

  extract_caffe_model ("cnn_resnet/ResNet-152-deploy.prototxt",
      "cnn_resnet/ResNet-152-model.caffemodel",
      d)

  for i in l:
    x = np.load (d+i)
    if x.ndim == 4:
      x = x.transpose ([2,3,1,0])
      np.save (d+i, x)

  x = np.random.randn(2048,output)
  x = x.astype ('float32')
  np.save (d+"fc_0.npy", x)
  x = np.random.randn(output)
  x = x.astype ('float32')
  np.save (d+"fc_1.npy", x)
