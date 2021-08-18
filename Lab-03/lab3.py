import numpy
import numpy as np
import cv2

from matplotlib import pyplot as plt


def threshold(gray_image: numpy.ndarray, thresholdValue, maxValue, thresh="BINARY"):
    threshold_image = gray_image.copy()
    height, width = gray_image.shape
    if thresh == "BINARY":
        v1 = maxValue; v2 = 0
    elif thresh == "BINARY_INV":
        v1 = 0; v2 = maxValue
    for i in range(height):
        for j in range(width):
            threshold_image[i,j] = v1 if threshold_image[i,j] >= thresholdValue else v2
    return threshold_image


def otsu_threshold(gray_image):
    hist = cv2.calcHist([gray_image],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.max()
    hist_norm_cumsum = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(255):
        p1,p2 = np.hsplit(hist_norm,[i])
        q1,q2 = hist_norm_cumsum[i],hist_norm_cumsum[255]-hist_norm_cumsum[i]
        b1,b2 = np.hsplit(bins,[i])
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2 
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    return thresh, threshold(gray_image, thresh, 255)


def band_threshold(gray_image: numpy.ndarray, low_threshold, high_threshold, maxValue):
    threshold_image = gray_image.copy()
    height, width = gray_image.shape
    for i in range(height):
        for j in range(width):
            threshold_image[i,j] = maxValue if threshold_image[i,j] >= low_threshold and threshold_image[i,j] <= high_threshold else 0
    return threshold_image


def semi_threshold(gray_image: numpy.ndarray, thresholdValue):
    threshold_image = gray_image.copy()
    height, width = gray_image.shape
    for i in range(height):
        for j in range(width):
            threshold_image[i,j] = threshold_image[i,j] if threshold_image[i,j] <= thresholdValue else 0
    return threshold_image


def bitwise_and(binary1, binary2):
    return binary1 & binary2


def multiLevel_threshold(image: numpy.ndarray, thresholdValue, maxValue):
    threshold_image = image.copy()
    height, width, channels = image.shape
    for i in range(height):
        for j in range(width):
            threshold_image[i,j][0] = maxValue if threshold_image[i,j][0] >= thresholdValue else 0
            threshold_image[i,j][1] = maxValue if threshold_image[i,j][1] >= thresholdValue else 0
            threshold_image[i,j][2] = maxValue if threshold_image[i,j][2] >= thresholdValue else 0
    return threshold_image


image = cv2.imread('Lab-03/lab(thresh).jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


## 1) Simple Thresholding
cv2.imshow('Grayscale image', threshold(gray_image, 85, 255))


## 2) Otsu Thresholding
# ret, imgf = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
# cv2.imshow("OTSU threshold 1 image", imgf)
# cv2.imshow('OTSU threshold 2 image', otsu_threshold(gray_image)[1])


## 3) Adaptive Thresholding
# imgf = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
# cv2.imshow('Adaptive threshold 1 image', imgf)


## 4) Band Thresholding
# cv2.imshow('Band threshold 1 image', band_threshold(gray_image, 95, 160, 255))
# binary1 = threshold(gray_image, 95, 255, "BINARY")
# binary2 = threshold(gray_image, 160, 255, "BINARY_INV")
# cv2.imshow('Band threshold 2 image', bitwise_and(binary1, binary2))


## 5) Semi Thresholding
# cv2.imshow('Semi threshold 1 image', semi_threshold(gray_image, 170))
# binary = threshold(gray_image, 170, 255, "BINARY_INV")
# cv2.imshow('Semi threshold 2 image', bitwise_and(gray_image, binary))


## 6) Multi-Level Thresholding
# cv2.imshow('Multi-Level threshold image', multiLevel_threshold(image, 60, 255))
# cv2.imshow('Original image', image)


cv2.imshow('Original grayscale image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

