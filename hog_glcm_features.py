import numpy as np
import cv2
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import exposure
from skimage.feature import greycomatrix, greycoprops

def extract_hog_features(img, img_size=(128,64), orientations=9, pixels_per_cell=(8,8), cells_per_block=(3, 3)):
    img = rgb2gray(img)
    img = resize(img, img_size,anti_aliasing=True)
    hog_features= hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                               cells_per_block=cells_per_block, block_norm='L2-Hys',
                               visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None)
    return hog_features

'''
Extracts the glcm statistical features.

params:
-------
img        np.array   rgb image to extract its features
distance   int        distance between the 2 pixels in the glcm matrix
angle      float      angle between the 2 pixels in the glcm matrix
verbose    boolean    visualize the values being extracted. By default: False
'''
def extract_glcm_features(img, distance=6, angle=-np.pi/9, verbose=False):
    img = (rgb2gray(img)*255).astype(np.uint8)
    glcm = greycomatrix(img, [distance], [angle], levels=256, normed=True, symmetric=True)
    contrast = greycoprops(glcm, 'contrast')[0][0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0][0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0][0]
    ASM = greycoprops(glcm, 'ASM')[0][0]
    energy = greycoprops(glcm, 'energy')[0][0]
    correlation = greycoprops(glcm, 'correlation')[0][0]

    if(verbose):
        print(contrast)
        print('---------')
        print(dissimilarity)
        print('---------')
        print(homogeneity)
        print('---------')
        print(ASM)
        print('---------')
        print(energy)
        print('---------')
        print(correlation)
        print('---------')
    glcm_features = [contrast, dissimilarity, homogeneity, ASM, energy, correlation]
    return glcm_features
