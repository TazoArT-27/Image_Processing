
#*Denoising, Smoothing and edge detection using opencv in Python
# Image smoothing, denoising
# Averaging, gaussian blurring, median, bilateral filtering
#OpenCV has a function cv2.filter2D(), which convolves whatever kernel we define with the image.

#***********Basically denoising and smoothing is the same thing*****************
#todo kernel is used basically for denoising. When we import an image it comes with a kernel. But we can define a kernel of our own and 
#todo it's size is small. It basically goes through the image and multiplies the image with the kernel and we see the output. 
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('Images/histSeg/BSE_Google_noisy.jpg', 1)
kernel = np.ones((5,5),np.float32)/25
#!This kernel image contains total 25 ones. I'm dividing this with 25 because I don't want to damage the energy of the image. 
#!So that the summation os the matrix elements remains one. 
filt_2D = cv2.filter2D(img,-1,kernel)    #Convolution using the kernel we provide
blur = cv2.blur(img,(5,5))   #Convolution with a normalized filter. Same as above for this example.
blur_gaussian = cv2.GaussianBlur(img,(5,5),0)  #Gaussian kernel is used.

#todo: median and bilateral filters are non-linear filtering. Non-linear filters are better than linear filters. 
#todo: Bilateral filters are best for denoising in OpenCV.
median_blur = median = cv2.medianBlur(img,5)  #Using kernel size 5. Better on edges compared to gaussian.
bilateral_blur = cv2.bilateralFilter(img,9,75,75)  #Good for noise removal but retain edge sharpness. 


#cv2.imshow("Original", img)
#cv2.imshow("2D filtered", filt_2D) #*image gets blurred
# cv2.imshow("Blur", blur)
# cv2.imshow("Gaussian Blur", blur_gaussian)
# cv2.imshow("Median Blur", median_blur)
# cv2.imshow("Bilateral", bilateral_blur)
# cv2.waitKey(0)          
# cv2.destroyAllWindows() 

#############################################################

#*Edge detection:


img1 = cv2.imread("Images/openCVBasicOperations/Neuron.jpg", 0)
edges = cv2.Canny(img1,100,200)   #Image, min and max values

# cv2.imshow("Original Image", img1)
# cv2.imshow("Canny", edges)

cv2.waitKey(0)          
cv2.destroyAllWindows() 
