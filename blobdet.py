# -*- coding: utf-8 -*-
"""Untitled36.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G6TukH-H31W7bpxpCqhruwNsJCRhRaQL
"""

from skimage.segmentation import watershed
import numpy as np
import cv2
from copy import deepcopy
from scipy import ndimage as ndi
import matplotlib.pyplot as plt


def count_big_bubbles(img1, min_size = 150):
  min_area = min_size
  gray_im = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(gray_im,180,255,0)  
  #plt.imshow(thresh)
  markers = ndi.label(thresh)[0]
  d = []
  ids, areas = np.unique(markers, return_counts=True)
  for id, area in zip(ids, areas):
    if area < min_area:
      d.append(id)
  #print(d)
  big_markers = deepcopy(markers)
  for id in d:
    big_markers[markers == id] = 0
  return np.unique(big_markers).shape[0]


