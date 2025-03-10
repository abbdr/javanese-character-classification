print('Importing dependencies...\n')

import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt
from PIL import Image
from KNN import KNN, train_test_split

aksara_jawa = ['ha','ca','ra','ka','na','da','ta','sa','wa','la','pa','dha','ja','ya','nya','ma','ga','ba','tha','nga',]

print('Loading images...\n')

path = f'D:\\DEV\\LatihanPython\\2025\\latihan\\aksara jawa\\train_white'
_img, _label = [], []
for file in os.listdir(path):
  filename = os.fsdecode(file)
  # print(filename)
  foo = Image.open(f'{path}\\{filename}')
  grayscale_image = foo.convert('L')
  data = np.asarray(grayscale_image).flatten()/255
  _img.append(data)
  for i in aksara_jawa:
    if i in filename:
      _label.append(i)

print('Splitting and shuffling data...\n')

train_img, test_img, train_label, test_label = train_test_split(_img, _label)

print('Calculating accuracy... (This may take a while !)\n')
clf.calculate_accuracy(test_img, test_label)

print('Classifying...\n')
clf = KNN()
clf.fit(train_img, train_label)
clf.predict(test_img[29])
