import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from skimage.exposure import match_histograms
from PIL import Image
from numpy import asarray

def canny_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(gray,(3,3),0)
    canny_img = cv2.Canny(image_blur,100,200)
    cv2.imshow('Canny image',canny_img)
    cv2.waitKey(0)

def histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist(gray,[0],None,[256],[0,256])
    plt.hist(gray.ravel(),256,[0,256])
    plt.show()

def histEq(image):
    cv2.imshow('Gray Scale Image', image)
    cv2.waitKey(0)
    equalized = cv2.equalizeHist(image)
    cv2.imwrite('equalized.jpg', equalized)
    eqImage = cv2.imread('equalized.jpg',0)
    hist = cv2.calcHist(eqImage, [0], None,[256], [0,256])
    plt.plot(hist)
    plt.show()
    equalizedImage = cv2.imread('equalized.jpg')
    cv2.imshow('Equalized Image',equalizedImage)
    cv2.waitKey(0)

def blackAndWhite(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img',gray)
    (threshold,binary) = cv2.threshold(gray,125,255,cv2.THRESH_BINARY)
    cv2.imwrite('bgimage.jpg',binary)
    bn = cv2.imread('bgimage.jpg')
    img = cv2.imshow('black and white',bn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def lowPass(image):
    kernel = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
    img = cv2.filter2D(image,-1,kernel/sum(kernel))
    cv2.imshow('Low pass Image',img)
    cv2.waitKey(0)

def highPass(image):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(image,-1,kernel)
    cv2.imshow('High Pass',img)
    cv2.waitKey(0)

def matched_histograms(image):
    ref = cv2.imread('D:\\SEMESTER\\SIXTH SEMESTER\\Computer Visualisation\\CG LAB\\cat.jpg')
    matched = match_histograms(image,ref)
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 2.5))

    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    ax2.imshow(ref, cmap=plt.cm.gray)
    ax2.set_title('Reference image')

    ax3.imshow(matched, cmap=plt.cm.gray)
    ax3.set_title('Matched image')

    for ax in (ax1, ax2, ax3):
        ax.axis('off')

    plt.show()

def neg_imag():
    img = Image.open('D:\\SEMESTER\\SIXTH SEMESTER\\Computer Visualisation\\CG LAB\\cat.jpg')
    data = asarray(img)
    data = abs(255 - 1 - data)
    image = Image.fromarray(data)
    image.save('NegativeTiger.jpg')

#image = cv2.imread('D:\SEMESTER\SIXTH SEMESTER\Computer Visualisation\CG LAB\cat.jpg')
neg_imag()
