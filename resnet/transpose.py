import numpy as np
import os

d = 'param_resnet/'
l = os.listdir(d)
for i in l:
  x = np.load (d+i)
  if x.ndim == 4:
    x = x.transpose ([2,3,1,0])
    np.save (d+i, x)
