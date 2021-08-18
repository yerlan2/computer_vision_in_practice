import numpy as np
import cv2


image = cv2.imread('LicensePlate.jpg')
height, width, channels = image.shape


def affine_transform(image, transformed_image, T):
    for i, row in enumerate(image):
        for j, col in enumerate(row):
            i_out, j_out, _ = T @ np.array([i, j, 1])
            transformed_image[int(i_out), int(j_out)] = image[i, j]


def nearest_neighbors(image, image_nn, T_inv):
    for i, row in enumerate(image_nn):
        for j, col in enumerate(row):
            i_out, j_out, _ = T_inv @ np.array([i, j, 1])
            image_nn[i, j] = image[int(i_out), int(j_out)]


T_scale = np.array([
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 1],
])
T_scale_inv = np.linalg.inv(T_scale)
image_nn = np.empty(( int(T_scale[0,0])*height, int(T_scale[1,1])*width, 3 ), dtype=np.uint8)
nearest_neighbors(image, image_nn, T_scale_inv)


near_img = cv2.resize(image,None, fx = 2, fy = 2, interpolation = cv2.INTER_NEAREST)
bilinear_img = cv2.resize(image,None, fx = 2, fy = 2, interpolation = cv2.INTER_LINEAR)
bicubic_img = cv2.resize(image,None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)


cv2.imshow('Original image', image)
cv2.imshow('Nearest neighbors image', image_nn)
cv2.imshow('OpenCV Nearest neighbors image', near_img)
cv2.imshow('OpenCV Bilinear image', bilinear_img)
cv2.imshow('OpenCV Bicubic image', bicubic_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

