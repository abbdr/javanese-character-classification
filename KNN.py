import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt
from PIL import Image

def train_test_split(_img, _label, split_ratio=0.95):
  l = np.arange(len(_img))
  np.random.shuffle(l)
  img, label = [], []
  for i in l:
    label.append(_label[i])
    img.append(_img[i])

  idx = int(split_ratio * len(img))

  train_img = img[:idx]
  train_label = label[:idx]

  test_img = img[idx:]
  test_label = label[idx:]

  return train_img, test_img, train_label, test_label


class KNN():
  def __init__(self, k=3):
      self.k = k

  def fit(self, X, y):
    self.train_img = X
    self.train_label = y

  def predict(self, X):
    k_nearest_labels, result = self._predict(X)
    print(f'Candidates : {k_nearest_labels}\n')
    print(f'Prediction : {result}\n')
    plt.imshow(self._show_image(X))
    plt.show()

  def calculate_accuracy(self, X, y):
    arr_hasil = []
    for i in range(len(X)):
        _, result = self._predict(X[i])
        if result == y[i]:
            arr_hasil.append(1)
        else:
            arr_hasil.append(0)

    _, count = np.unique(arr_hasil, return_counts=True)
    loss = (count[0]/np.sum(count))*100
    acc = (count[1]/np.sum(count))*100
    print('Accuracy: ', acc, ' %')
    print('Loss: ', loss, ' %')

  
  def _predict(self, X):
    testing = X
    hasil = np.array([np.sqrt(np.sum((training - testing) ** 2)) for training in self.train_img])
    k_indices = np.argsort(hasil)[:self.k]
    k_nearest_labels = [self.train_label[i] for i in k_indices]
    result = self._numpy_mode(k_nearest_labels)

    return k_nearest_labels, result


  def _numpy_mode(self, data):
    (sorted_data, idx, counts) = np.unique(data, return_index=True, return_counts=True)
    index = idx[np.argmax(counts)]

    return data[index]

  def _show_image(self, data):
    arr_img = []
    idx = np.arange(0,224**2+1,224)
    for i in range(len(idx)-1):
      arr_img.append(data[idx[i]:idx[i+1]])
    img = Image.fromarray(np.array(arr_img).astype('uint8'))
    return img