
#*Histogram Equalization and Contrast Limited Adaptive Histogram Equalization(CLHAE)

#?WHAT IS CLAHE?
#*CLAHE is a modification of the traditional histogram equalization method that aims to 
#*enhance local contrast in an image while avoiding over-amplification of noise.

#?WHAT IS CONTRAST OF AN IMAGE?
#*Contrast is the difference between the darker and the brighter part of the image. 
#*As you increase this difference, it gives you the impression of stronger color and texture(deepens the shadows and brighten the highlights) and
#*as you decrease the difference your image will look like little bit flat. 

#?WHAT IS CLIPLIMIT IN CLAHE?
#*The cliplimit parameter determines the threshold for contrast limiting in CLAHE. 
#*It limits the amount of contrast enhancement applied to each pixel neighborhood. 
#*The value of cliplimit is typically specified between 0 and 1. 
#*A higher cliplimit value allows more contrast enhancement, but it also increases the risk of over-amplifying noise and artifacts. 
#*On the other hand, a lower cliplimit value reduces the amount of enhancement and helps preserve image details.

#?WHAT IS TILESIZE IN CLAHE?
#*The tilesize parameter defines the size of the local neighborhood or tile over which the contrast equalization is performed.
#*The image is divided into smaller non-overlapping tiles or blocks, and histogram equalization is applied individually to each tile. 
#*The tilesize determines the dimensions of these tiles. 
#*Smaller tile sizes capture finer local details but may result in a "patchwork" effect, while larger tile sizes smooth out local variations but may lose some details. 
#*The appropriate tilesize depends on the characteristics of the image and the desired level of local contrast enhancement.

#Image enhancements: 
# Sometimes microscope images lack contrast, they appear to be washed out but they still contain information.
# (Show scratch assay and alloy images)
# We can mathematically process these images and make them look good,
#more importantly, get them ready for segmentation
#
#Histogram equalization is a good way to stretch the histogram and thus improve the image.  


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("Images/randomWalker/Alloy_noisy.jpg", 0)
equ = cv2.equalizeHist(img)

plt.hist(equ.flat, bins=100, range=(0,100))

# cv2.imshow("Original Image", img);
# cv2.imshow("Equalized", equ)


#Histogram Equalization considers the global contrast of the image, may not give good results.
#Adaptive histogram equalization divides images into small tiles and performs hist. eq.
#Contrast limiting is also applied to minimize aplification of noise.
#Together the algorithm is called: Contrast Limited Adaptive Histogram Equalization (CLAHE)

# Start by creating a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  #Define tile size and clip limit. 
cl1 = clahe.apply(img)

# cv2.imshow("CLAHE", cl1)

cv2.waitKey(0)          
cv2.destroyAllWindows() 


######################################################
###################################################
#Image thresholding

# import cv2
# import matplotlib.pyplot as plt

img1 = cv2.imread("Images/randomWalker/Alloy_noisy.jpg", 0)

#Adaptive histogram equalization using CLAHE to stretch the histogram. 
#Contrast Limited Adaptive Histogram Equalization covered in the previous tutorial. 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  #Define tile size and clip limit. 
clahe_img = clahe.apply(img1)
# plt.hist(clahe_img.flat, bins =100, range=(0,255))

#Thresholding. Creates a uint8 image but with binary values.
#Can use this image to further segment.
#First argument is the source image, which should be a grayscale image.
#Second argument is the threshold value which is used to classify the pixel values. 
#Third argument is the maxVal which represents the value to be given to the thresholded pixel.

ret,thresh1 = cv2.threshold(clahe_img,185,150,cv2.THRESH_BINARY)  #All thresholded pixels in grey = 150
#*here 185 threshold value comes from the histogram. And all the values under the threshold will be of 150 intensity of gray.
ret,thresh2 = cv2.threshold(clahe_img,185,255,cv2.THRESH_BINARY_INV) # All thresholded pixels in white


# cv2.imshow("Original", img1)
# cv2.imshow("Binary thresholded", thresh1)
# cv2.imshow("Inverted Binary thresholded", thresh2)
cv2.waitKey(0)          
cv2.destroyAllWindows() 

############################################
#OTSU Thresholding, binarization
#TODO: In normal thresholding we were looking at the histogram and guessing the value of segmentation. 
#todo: but in OTSU thresholding we do not do that, and it is better than the previous thresholding. 
#todo: OTSU calculates the threshold value automatically.  
# import cv2
# import matplotlib.pyplot as plt

img2 = cv2.imread("Images/randomWalker/Alloy_noisy.jpg", 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  #Define tile size and clip limit. 
clahe_img = clahe.apply(img2)

plt.hist(clahe_img.flat, bins =100, range=(0,255))

# binary thresholding
ret1,th1 = cv2.threshold(clahe_img,185,200,cv2.THRESH_BINARY)

# Otsu's thresholding, automatically finds the threshold point. 
#Compare wth above value provided by us (185)
ret2,th2 = cv2.threshold(clahe_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# cv2.imshow("Otsu", th2)
cv2.waitKey(0)          
cv2.destroyAllWindows() 

# If working with noisy images
# Clean up noise for better thresholding
# Otsu's thresholding after Gaussian filtering. Canuse median or NLM for beteer edge preserving

# import cv2
# import matplotlib.pyplot as plt

img3 = cv2.imread("Images/randomWalker/Alloy_noisy.jpg", 0)

blur = cv2.GaussianBlur(clahe_img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


plt.hist(blur.flat, bins =100, range=(0,255))
# cv2.imshow("OTSU Gaussian cleaned", th3)
cv2.waitKey(0)          
cv2.destroyAllWindows() 