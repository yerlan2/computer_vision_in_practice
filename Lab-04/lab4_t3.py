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

input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
output_pts = np.float32([
    [0, 0],
    [0, maxHeight - 1],
    [maxWidth - 1, maxHeight - 1],
    [maxWidth - 1, 0]
])


def create_perspective_transform_matrix(src, dst):
    in_matrix = []
    for (x, y), (X, Y) in zip(src, dst):
        in_matrix.extend([
            [x, y, 1, 0, 0, 0, -X * x, -X * y],
            [0, 0, 0, x, y, 1, -Y * x, -Y * y],
        ])
    A = np.matrix(in_matrix, dtype=np.float)
    B = np.array(dst).reshape(8)
    af = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.append(np.array(af).reshape(8), 1).reshape((3, 3))


image = cv2.imread('LicensePlate.jpg')

M = cv2.getPerspectiveTransform(input_pts, output_pts)
dst = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

plt.subplot(121),plt.imshow(image[:,:,::-1]),plt.title('Input')
plt.subplot(122),plt.imshow(dst[:,:,::-1]),plt.title('Output')
plt.show()


M = create_perspective_transform_matrix(input_pts, output_pts)
dst = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

plt.subplot(121),plt.imshow(image[:,:,::-1]),plt.title('Input')
plt.subplot(122),plt.imshow(dst[:,:,::-1]),plt.title('Output')
plt.show()

