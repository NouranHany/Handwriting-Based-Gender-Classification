## File containing helper - common functions used in the project

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import glob

'''
Plots the given image

params:
-------
img:         image to be plotted.
normalized:  whether the image values ranges from 0-255 or 0-1. default
'''
def show_image(img, normalized=False):
    if (normalized):
        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.show()

'''
Plot an array of images as a grid of images

params:
-------
figsize  tuble  width and height of each figure.
rows     int    count of grid rows.
cols     int    count of grid cols.
imgs     list   list of images, each image is stored as a numpy array. 
'''
def draw_grid(figsize=(10,10), rows=5,cols=5,imgs=[]):
    fig = plt.figure(figsize=figsize)

    for i in range(len(imgs)):
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(imgs[i])
    plt.show()


'''
Read all images placed at the given directory.

params:
-------
dir_path  string  path of directory that contain images to be read.

return:
------
imgs      list    list of all images read from the directory. each image is read as a numpy array.

'''
def read_imgs(dir_path='.'):
    paths = glob.glob(dir_path+'/**')
    imgs = []
    for img_path in paths:
        imgs.append(cv2.imread(img_path))
    return imgs