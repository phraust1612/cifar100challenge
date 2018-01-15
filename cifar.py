import pickle
import numpy as np

data_len = 50000
test_len = 10000

def cifar10train (i:int, j:int):
  data = np.array ([], dtype=np.float32)
  data = data.reshape ([0,32,32,3])
  label = np.array ([], dtype=np.int32)
  bi = i // 10000 + 1
  bj = (j - 1) // 10000 + 1
  i = i % 10000
  j = (j - 1) % 10000 + 1

  for fi in range (bi, bj):
    tmp, tmp2 = GetDataFromFile ("data/data_batch_"+str(fi), i, 10000, b'labels')
    data = np.concatenate ([data, tmp])
    label = np.concatenate ([label, tmp2], -1)
    i = 0

  tmp, tmp2 = GetDataFromFile ("data/data_batch_"+str(bj), i, j, b'labels')
  data = np.concatenate ([data, tmp])
  label = np.concatenate ([label, tmp2], -1)

  del (tmp)
  del (tmp2)

  return data, label

def cifar10test (i:int, j:int):
  return GetDataFromFile ("data/test_batch", i, j, b'labels')

def cifar100train (i:int, j:int):
  return GetDataFromFile ("data/train", i, j, b'fine_labels')

def cifar100test (i:int, j:int):
  return GetDataFromFile ("data/test", i, j, b'fine_labels')

def GetDataFromFile (name:str, i:int, j:int, labelkey:bytes):
  with open (name, "rb") as f:
    dic = pickle.load (f, encoding="bytes")
    data = np.array (dic[b'data'][i:j], dtype=np.float32)
    data = data.reshape ([j-i,3,32,32])
    data = data.transpose ([0,2,3,1])
    label = np.array (dic[labelkey][i:j], dtype=np.int32)

  del (dic)
  return data, label
