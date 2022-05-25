import cv2
import numpy as np
import matplotlib.pyplot as pp
from skimage.measure import find_contours
from utilities import *


def character_size(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    sizes = np.zeros(0)
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        size = w*h
        if (size<500000):
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            sizes = np.append(sizes, size)
    if(len(sizes)) != 0:
        return np.mean(sizes)
    else:
        return 0

def character_size_2(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    sizes = np.zeros(0)
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        size = w*h
        if (size > 100 ) and (size<10000) :
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,(size / 10000) * 255,0), 2)
            sizes = np.append(sizes, size)
    if(len(sizes)) != 0:
        return np.mean(sizes)
    else:
        return 0

