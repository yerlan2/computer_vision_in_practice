import numpy as np
import cv2

import matplotlib.pyplot as plt


pt_A = [27, 67]
pt_B = [24, 136]
pt_C = [253, 73]
pt_D = [256, 8]

width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
maxWidth = max(int(width_AD), int(width_BC))

height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
maxHeight = max(int(height_AB), int(height_CD))

input_pts = np.float32([pt_A, pt_B, pt_C])
output_pts = np.float32([
    [0, 0],
    [0, maxHeight - 1],
    [maxWidth - 1, maxHeight - 1],
])


image = cv2.imread('LicensePlate.jpg')

height, width, channels = image.shape
M = cv2.getAffineTransform(input_pts, output_pts)
dst = cv2.warpAffine(image, M, (width, height))

plt.subplot(121),plt.imshow(image[:,:,::-1]),plt.title('Input')
plt.subplot(122),plt.imshow(dst[:,:,::-1]),plt.title('Output')
plt.show()

