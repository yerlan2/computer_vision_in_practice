import numpy as np
import cv2


image = cv2.imread('LicensePlate.jpg')

deg = np.deg2rad(15)
height, width, channels = image.shape


def affine_transform(image, transformed_image, T):
    for i, row in enumerate(image):
        for j, col in enumerate(row):
            i_out, j_out, _ = T @ np.array([i, j, 1])
            transformed_image[int(i_out), int(j_out)] = image[i, j]


def nearest_neighbors(image, transformed_image, T_inv):
    for i, row in enumerate(transformed_image):
        for j, col in enumerate(row):
            i_out, j_out, _ = T_inv @ np.array([i, j, 1])
            transformed_image[i, j] = image[round(i_out), round(j_out)]


# Skewing
T_skewing = np.array([
    [1, np.tan(deg), -int( width * np.tan(deg) ) if deg < 0 else 0],
    [0, 1, 0],
    [0, 0, 1],
])
skewing_transformed_image = 255-np.zeros(( int(T_skewing[0,0])*height + int(width * np.abs(np.tan(deg))), int(T_skewing[1,1])*width, 3 ), dtype=np.uint8)
affine_transform(image, skewing_transformed_image, T_skewing)

# Panoramic
T_panoramic = np.array([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 1],
])
T_panoramic_inv = np.linalg.inv(T_panoramic)

# Skewing and Panoramic
T = np.array([
    [1, np.tan(deg), -int( width * np.tan(deg) ) if deg < 0 else 0],
    [0, 2, 0],
    [0, 0, 1],
])
skewing_panoramic_transformed_image = 255-np.zeros(( int(T[0,0])*height + int(width * np.abs(np.tan(deg))), int(T[1,1])*width, 3 ), dtype=np.uint8)
affine_transform(image, skewing_panoramic_transformed_image, T)


image_nn = np.empty(skewing_panoramic_transformed_image.shape, dtype=np.uint8)
nearest_neighbors(skewing_transformed_image, image_nn, T_panoramic_inv)


cv2.imshow('Original image', image)
cv2.imshow('Skewing and Panoramic transformed image', skewing_panoramic_transformed_image)
cv2.imshow('Nearest neighbors image', image_nn)
cv2.waitKey(0)
cv2.destroyAllWindows()

