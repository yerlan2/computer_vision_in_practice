import numpy
import cv2

from matplotlib import pyplot as plt


def imreads(imagePathList):
    imageList = []
    for imagePath in imagePathList:
        imageList.append(cv2.imread(imagePath))
    return imageList


def myPlotHistogram(histogram: numpy.ndarray):
    plt.figure()
    plt.title("Color image histogram")
    plt.xlabel('Intensity level')
    plt.ylabel('Intensity frequency')
    plt.xlim([0, 256])
    plt.plot(histogram[:, 0], 'b')
    plt.plot(histogram[:, 1], 'g')
    plt.plot(histogram[:, 2], 'r')
    plt.show()


def myHistogram(imageList: list):
    histogramList = []
    for image in imageList:
        height, weight, channels = image.shape
        histogram = numpy.zeros([256, channels], numpy.int32)
        for x in range(height):
            for y in range(weight):
                for c in range(channels):
                    histogram[image[x,y,c], c] += 1
        histogramList.append(histogram)
    return histogramList


def myNormalize(histList: list, alpha=None, beta=None, norm_type=None):
    normalizedHistogramList = []
    for hist in histList:
        if norm_type == "NORM_L1":
            normalizedHistogramList.append(hist/hist.sum(axis=0))
        elif norm_type == "NORM_MINMAX":
            a, b = hist.min(), hist.max()
            normalizedHistogramList.append((hist-a)/(b-a)*(beta-alpha) + alpha)
        else:
            normalizedHistogramList.append(hist/numpy.sqrt((hist ** 2).sum(axis=0)))
    return normalizedHistogramList


def myCompareHist(hist1, hist2):
    compareHist = numpy.sum( (hist1 - numpy.mean(hist1, axis=0)) * (hist2 - numpy.mean(hist2, axis=0)), axis=0 ) \
        / numpy.sqrt( numpy.sum((hist1 - numpy.mean(hist1, axis=0))**2, axis=0) * numpy.sum((hist2 - numpy.mean(hist2, axis=0))**2, axis=0) )
    return compareHist.mean().round(5)


def putTextImage(image, text):
    return cv2.putText(image, f"{text}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, .5, [0,0,255])


mainImage = cv2.imread('Lab-02/images/lab2_1.jpg')
imagePathList = [
    'Lab-02/images/lab2_1copy.jpg', 
    'Lab-02/images/lab2_2.jpg', 
    'Lab-02/images/lab2_2copy.jpg', 
    'Lab-02/images/lab2_3.jpg',
    'Lab-02/images/lab2_3copy.jpg', 
    'Lab-02/images/lab2_4.jpg',
    'Lab-02/images/lab2_4copy.jpg', 
    'Lab-02/images/lab2_5.jpg',
    'Lab-02/images/lab2_6.jpg',
    'Lab-02/images/lab2_6copy.jpg', 
    'Lab-02/images/lab2_7.jpg',
    'Lab-02/images/lab2_8.jpg',
    'Lab-02/images/lab2_8copy.jpg', 
    'Lab-02/images/lab2_9.jpg',
    'Lab-02/images/lab2_10.jpg',
    'Lab-02/images/lab2_10copy.jpg',
    'Lab-02/images/lab2_11.jpg',
    'Lab-02/images/lab2_11copy.jpg',
    'Lab-02/images/lab2_12.jpg',
    'Lab-02/images/lab2_13.jpg',
    'Lab-02/images/lab2_13copy.jpg',
    'Lab-02/images/lab2_14.jpg',
    'Lab-02/images/lab2_14copy.jpg',
    'Lab-02/images/lab2_15.jpg',
    'Lab-02/images/lab2_15copy.jpg',
    'Lab-02/images/lab2_16.jpg',
    'Lab-02/images/lab2_17.jpg',
    'Lab-02/images/lab2_17copy.jpg',
    'Lab-02/images/lab2_18.jpg',
    'Lab-02/images/lab2_19.jpg',
    'Lab-02/images/lab2_19copy.jpg',
    'Lab-02/images/lab2_20.jpg',
    'Lab-02/images/lab2_20copy.jpg',
    'Lab-02/images/lab2_21.jpg',
    'Lab-02/images/lab2_21copy.jpg',
    'Lab-02/images/lab2_22.jpg',
    'Lab-02/images/lab2_22copy.jpg',
    'Lab-02/images/lab2_23.jpg',
    'Lab-02/images/lab2_23copy.jpg',
    'Lab-02/images/lab2_24.jpg',
]
imageList = imreads(imagePathList)

mainHistList = myHistogram([mainImage])
histList = myHistogram(imageList)

normalizedMainHistList = myNormalize(mainHistList)
normalizedHistList = myNormalize(histList)

cv2.imshow('Main image', mainImage)
i = 0
for image, normalizedHist in zip(imageList, normalizedHistList):
    i += 1
    metric_val = myCompareHist(normalizedMainHistList[0], normalizedHist)
    print(f"Image {i:2} = {metric_val}")
    cv2.imshow(f'Image {i}', putTextImage(image, metric_val))
cv2.waitKey(0)
cv2.destroyAllWindows()

