import numpy
import cv2


def changeQuantisationGrey(image: numpy.ndarray, num_bits: int):
    convImage = image.copy()
    if num_bits >= 1 and num_bits <= 8:
        height, width, channels = image.shape
        mask = 0xFF << (8 - num_bits)
        for i in range(height):
            for j in range(width):
                convImage[i,j] = numpy.clip(0.299*image[i,j][2] + 0.587*image[i,j][1] +  0.114*image[i,j][0], 0, 255)
                convImage[i,j] = convImage[i,j] & mask
    cv2.imshow("Images", numpy.hstack([ image, convImage ]))
    # cv2.imshow('Original image', image)
    # cv2.imshow('Gray image', convImage)
    # cv2.imshow('OpenCV\'s gray image', cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    cv2.waitKey(0)


def cmy(image: numpy.ndarray):
    convImage = image.copy()
    height, width, channels = image.shape
    for i in range(height):
        for j in range(width):
            convImage[i,j] = 255-image[i,j]
    cv2.imshow("Images", numpy.hstack([ image, convImage ]))
    # cv2.imshow('Original image', image)
    # cv2.imshow('CMY image', convImage)
    cv2.waitKey(0)

def yuv(image: numpy.ndarray):
    convImage = image.copy()
    height, width, channels = image.shape
    for i in range(height):
        for j in range(width):
            Y = numpy.clip(0.299*image[i,j][2] + 0.587*image[i,j][1] +  0.114*image[i,j][0], 0, 255)
            convImage[i,j][0] = Y
            convImage[i,j][1] = numpy.clip(0.492*(image[i,j][0] - Y) + 128, 0, 255) # U
            convImage[i,j][2] = numpy.clip(0.877*(image[i,j][2] - Y) + 128, 0, 255) # V
    cv2.imshow("Images", numpy.hstack([ image, convImage ]))
    # cv2.imshow('Original image', image)
    # cv2.imshow('CMY image', convImage)
    cv2.imshow('OpenCV\'s YUV image', cv2.cvtColor(image, cv2.COLOR_BGR2YUV))
    cv2.waitKey(0)


def hls(image: numpy.ndarray):
    convImage = image.copy()
    height, width, channels = image.shape
    cv2.imshow('Original image', image)
    cv2.imshow('Image', cv2.cvtColor(image, cv2.COLOR_BGR2HLS))
    cv2.waitKey(0)


def add_salt_and_pepper_noise(img, noise_amount=0.1):
    noise_image = img.copy()
    noise_points = int(noise_image.size * noise_amount)
    for n in range(noise_points):
        pixel = tuple(numpy.random.randint(noise_image.shape))
        noise_image[pixel] = 255 if numpy.random.randint(256) % 2 == 1 else 0
    return noise_image

def add_gaussian_noise(image_in: numpy.ndarray, noise_sigma=0.5):
    height, width, channels = image_in.shape
    noise_image = \
        numpy.random.normal(0, noise_sigma, image_in.size). \
        reshape(height, width, channels).astype('uint8')
    noise_image = cv2.addWeighted(image_in, 1., noise_image, 1., 0.)
    return noise_image


image = cv2.imread('Lab-01/labka (1).jpg')


# changeQuantisationGrey(image, 2)
# changeQuantisationGrey(image, 4)
# changeQuantisationGrey(image, 6)
# cmy(image)
# yuv(image)
# hls(image)

# salt_and_pepper_noise_image = add_salt_and_pepper_noise(image)
# cv2.imshow('"Salt & Pepper" noise image', salt_and_pepper_noise_image)
# gaussian_noise_image = add_gaussian_noise(image)
# cv2.imshow('Gaussian noise image', gaussian_noise_image)
# cv2.waitKey(0)

# median = cv2.medianBlur(salt_and_pepper_noise_image, 5)
# cv2.imshow('Median Blur "Salt & Pepper" noise image', median)
# median = cv2.medianBlur(gaussian_noise_image, 5)
# cv2.imshow('Median Blur Gaussian noise image', median)
# cv2.waitKey(0)

