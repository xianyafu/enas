import os
import sys
#import cPickle as pickle
import pickle
import numpy as np
import tensorflow as tf
class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()

def _read_data(data_path, train_files):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
  for file_name in train_files:
    print(file_name)
    full_name = os.path.join(data_path, file_name)
    if os.path.getsize(full_name) <= 0:
      print(full_name)
    with open(full_name,'rb') as finp:
      data = pickle.load(finp,encoding='iso-8859-1')
      batch_images = data["data"].astype(np.float32) / 255.0
      batch_labels = np.array(data["labels"], dtype=np.int32)
      images.append(batch_images)
      labels.append(batch_labels)
  images = np.concatenate(images, axis=0)
  labels = np.concatenate(labels, axis=0)
  images = np.reshape(images, [-1, 3, 32, 32])
  images = np.transpose(images, [0, 2, 3, 1])

  return images, labels


def read_data(data_path, num_valids=5000):
  print("-" * 80)
  print("Reading data")

  images, labels = {}, {}

  train_files = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
  ]
  test_file = [
    "test_batch",
  ]
  images["train"], labels["train"] = _read_data(data_path, train_files)

  if num_valids:
    images["valid"] = images["train"][-num_valids:]
    labels["valid"] = labels["train"][-num_valids:]

    images["train"] = images["train"][:-num_valids]
    labels["train"] = labels["train"][:-num_valids]
  else:
    images["valid"], labels["valid"] = None, None

  images["test"], labels["test"] = _read_data(data_path, test_file)

  print("Prepropcess: [subtract mean], [divide std]")
  mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
  std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

  print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
  print("std: {}".format(np.reshape(std * 255.0, [-1])))

  images["train"] = (images["train"] - mean) / std
  if num_valids:
    images["valid"] = (images["valid"] - mean) / std
  images["test"] = (images["test"] - mean) / std

  return images, labels

def _read_data_by_order(data_path, train_files, num):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
  for file_name in train_files:
    print(file_name)
    full_name = os.path.join(data_path, file_name)
    if os.path.getsize(full_name) <= 0:
      print(full_name)
    with open(full_name,'rb') as finp:
      data = pickle.load(finp)
      batch_images = data["data"].astype(np.float32) / 255.0
      batch_labels = np.array(data["fine_labels"], dtype=np.int32)
      for i in range(batch_labels.shape[0]):
        if batch_labels[i] in range(num, num+10):
          images.append(batch_images[i])
          labels.append(batch_labels[i])
  images = np.concatenate(images, axis=0)
  labels = np.array(labels)
  #labels = np.concatenate(labels, axis=0)
  images = np.reshape(images, [-1, 3, 32, 32])
  images = np.transpose(images, [0, 2, 3, 1])

  return images, labels

def _read_data_by_order_v1(data_path, train_files, num):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
  for file_name in train_files:
    print(file_name)
    full_name = os.path.join(data_path, file_name)
    if os.path.getsize(full_name) <= 0:
      print(full_name)
    with open(full_name,'rb') as finp:
      data = pickle.load(finp)
      batch_images = data["data"].astype(np.float32) / 255.0
      batch_labels = np.array(data["fine_labels"], dtype=np.int32)
      for i in range(batch_labels.shape[0]):
        if batch_labels[i] in range(0, num+10):
          images.append(batch_images[i])
          labels.append(batch_labels[i])
  images = np.concatenate(images, axis=0)
  labels = np.array(labels)
  #labels = np.concatenate(labels, axis=0)
  images = np.reshape(images, [-1, 3, 32, 32])
  images = np.transpose(images, [0, 2, 3, 1])

  return images, labels



def read_data_by_order(data_path, num_valids=500):
  print("-" * 80)
  print("Reading data")

  total_images, total_labels = {}, {}

  train_files = [
    "train"
  ]
  test_file = [
    "test",
  ]
  for i in range(0, 10):
    images, labels = {}, {}
    images["train"], labels["train"] = _read_data_by_order(data_path, train_files, i*10)
    if num_valids:
      images["valid"] = images["train"][-num_valids:]
      labels["valid"] = labels["train"][-num_valids:]
      
      images["train"] = images["train"][:-num_valids]
      labels["train"] = labels["train"][:-num_valids]
    else:
      images["valid"], labels["valid"] = None, None
 
    images["test"], labels["test"] = _read_data_by_order_v1(data_path, test_file, i*10)
    print(i)
    print("Prepropcess: [subtract mean], [divide std]")
    mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
    std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)
 
    print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
    print("std: {}".format(np.reshape(std * 255.0, [-1])))
 
    #images["train"] = (images["train"] - mean) / std
    #if num_valids:
    #  images["valid"] = (images["valid"] - mean) / std
    #images["test"] = (images["test"] - mean) / std
    total_images[i]=images
    total_labels[i]=labels
    print(total_images[i]["train"].shape)
    print(total_labels[i]["train"].shape)
  return total_images, total_labels

