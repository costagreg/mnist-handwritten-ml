import cv2
import math
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy import ndimage
from utils import ValueInvert


# TO-DO: Refactor this with np.nonzero??
def find_center_image(img):
    left = 0
    right = img.shape[1] - 1

    empty_left = True
    empty_right = True

    for col in range(int(img.shape[1])):
        if empty_left == False and empty_right == False:
            break
        for row in range(img.shape[0] - 1):
            if img[row, col] > 0 and empty_left == True:
                empty_left = False
                left = col

            if img[row, img.shape[1] - col - 1] > 0 and empty_right == True:
                empty_right = False
                right = img.shape[1] - col

    top = 0
    bottom = img.shape[0] - 1

    empty_top = True
    empty_bottom = True

    for row in range(int(img.shape[0])):
        if empty_top == False and empty_bottom == False:
            break

        for col in range(img.shape[1] - 1):
            if img[row, col] > 0 and empty_top == True:
                empty_top = False
                top = row

            if img[img.shape[0] - row - 1, col] > 0 and empty_bottom == True:
                empty_bottom = False
                bottom = img.shape[0] - row

    return top, right, bottom, left


def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def process_image(img):
    img = ValueInvert(img)
    img = cv2.resize(img, (28, 28))
    (thresh, gray) = cv2.threshold(img, 128,
                                   255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    top, right, bottom, left = find_center_image(img)
    cropped_img = img[top:bottom, left:right]

    rows, cols = cropped_img.shape

    # resize 20x20 keeping ratio
    if rows > cols:
        rows = 20
        factor = cols/rows
        cols = int(round(rows*factor))
    else:
        cols = 20
        factor = rows/cols
        rows = int(round(cols*factor))

    gray = cv2.resize(cropped_img, (cols, rows))
    # plt.imshow(gray)
    # plt.show()
    colsPadding = (int(math.ceil((28-cols)/2.0)),
                   int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),
                   int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

    shiftx, shifty = getBestShift(gray)
    shifted = shift(gray, shiftx, shifty)
    gray = shifted

    return gray
