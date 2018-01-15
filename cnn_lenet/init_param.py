import numpy as np
import math

param_dir = "cnn_lenet/param_lenet/"

def init_lenet (output):
  if output != 100 and output != 10:
    return -1
  W = np.random.randn (3,3,3,32)
  W = W / (math.sqrt(3) * 12.0)
  W = W.astype ('float32')
  np.save (param_dir + "conv1_0.npy", W)

  b = np.random.randn (32)
  b = b / 4.0
  b = b.astype ('float32')
  np.save (param_dir + "conv1_1.npy", b)

  W = np.random.randn (3,3,32,64)
  W = W / 96.0
  W = W.astype ('float32')
  np.save (param_dir + "conv2_0.npy", W)

  b = np.random.randn (64)
  b = b / (math.sqrt(2) * 4.0)
  b = b.astype ('float32')
  np.save (param_dir + "conv2_1.npy", b)

  W = np.random.randn (8*8*64, 1024)
  W = W * math.sqrt(2) / 2048.0
  W = W.astype ('float32')
  np.save (param_dir + "fc1_0.npy", W)

  b = np.random.randn (1024)
  b = b * math.sqrt(2) / 32.0
  b = b.astype ('float32')
  np.save (param_dir + "fc1_1.npy", b)

  W = np.random.randn (1024,output)
  W = W * math.sqrt(2) / 320.0
  W = W.astype ('float32')
  np.save (param_dir + "fc2_0.npy", W)

  b = np.random.randn (output)
  b = b * math.sqrt(2) / 10.0
  b = b.astype ('float32')
  np.save (param_dir + "fc2_1.npy", b)

  return 0
