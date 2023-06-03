
#*Key points, detectors and descriptors in openCV

#!In OpenCV, keypoints, detectors, and descriptors are fundamental components used for feature detection, 
#!description, and matching in computer vision and image processing tasks. 

#?WHAT IS IMAGE REGISTRATION?
#*Image registration is the process of aligning two or more images of the same scene or object taken at different times, 
#*from different viewpoints, or by different imaging modalities. 
#*The goal of image registration is to find a spatial transformation that brings the images into a common coordinate system, 
#*allowing for direct comparison or combination of the images.

#?WHAT IS KEYPOINTS IN OPENCV?
#*Keypoints represent distinctive and localized feature points or regions within an image. 
#*They are typically identified based on their unique characteristics, such as corners, edges, or blobs. 
#*Keypoints are essential for various tasks, including feature matching, object recognition, image registration, and tracking.

#?WHAT IS DETECTORS IN OPENCV?
#*Detectors are algorithms or methods used to detect keypoints in an image. 

#?WHAT IS DESCRIPTORS IN OPENCV?
#*Descriptors are numerical representations or feature vectors associated with each keypoint. 
#*They capture the local image information surrounding the keypoint and describe its distinctive characteristics. 
#*Descriptors are computed to encode the appearance, texture, or other relevant information of the keypoints. 



"""What are features, detectors, and keypoints?
Features in an image are unique regions that the computer can easily tell apart.
Corners are good examples of features. Finding these unique features is called feature detection.
Once features are detected we need to find similar ones in a different image.
This means we need to describe the features. 
Once you have the features and its description, you can find same features 
in all images and align them, stitch them or do whatever you want.

Harris corner detector is a good example of feature detector. 

Keypoints are the same thing as points of interest. 
They are spatial locations, or points in the image that define what is interesting or what stand out in the image.
The reason why keypoints are special is because no matter how the image changes...
whether the image rotates, shrinks/expands, is translated, or distorted
you should be able to find the same keypoints in this modified image when comparing with the original image.

Harris corner detector detects corners. 
FAST: Features from Accelerated Segment Test - also detects corners


Each keypoint that you detect has an associated descriptor that accompanies it.
SIFT, SURF and ORB all detect and describe the keypoints.

Descriptors are primarily concerned with both the scale and the orientation of the keypoint.

e.g. run ORB and draw keypoints with rich keypoints.
some of these points have a different circle radius. These deal with scale.

The larger the "circle", the larger the scale was that the point was detected at.
Also, there is a line that radiates from the centre of the circle to the edge. 
This is the orientation of the keypoint, which we will cover next

Usually when we want to detect keypoints, we just take a look at the locations. 
However, if you want to match keypoints between images, 
then you definitely need the scale and the orientation to facilitate this.

"""

#HARRIS corner
#

import cv2
import numpy as np


img = cv2.imread('Images/grains/grains.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)  #Harris works on float32 images. 

#Input parameters
# image, block size (size of neighborhood considered), ksize (aperture parameter for Sobel), k
harris = cv2.cornerHarris(gray,2,3,0.04)  

# Threshold for an optimal value, it may vary depending on the image.
img[harris>0.01*harris.max()]=[255,0,0]    # replace these pixels with blue

# cv2.imshow('Harris Corners',img)
# cv2.waitKey(0)

#############################################

# Shi-Tomasi Corner Detector & Good Features to Track
# In opencv it is called goodfeaturestotrack


# import cv2
# import numpy as np

img1 = cv2.imread('images/grains/grains.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#input image, #points, quality level (0-1), min euclidean dist. between detected points
corners = cv2.goodFeaturesToTrack(gray,50,0.05,10)
corners = np.int0(corners)   #np.int0 is int64
#todo 'corners' is all the detected points in x and y coordinates.

for i in corners:
    x,y = i.ravel()   # Ravel Returns a contiguous flattened array.
#    print(x,y)
    cv2.circle(img,(x,y),3,255,-1)  #Draws circle (Img, center, radius, color, etc.)

# cv2.imshow('Corners',img)
# cv2.waitKey(0)

#####################################
#SIFT and SURF - do not work in opencv 3
#SIFT stands for scale invariant feature transform
#####################################
# FAST
# Features from Accelerated Segment Test
# High speed corner detector
# FAST is only keypoint detector. Cannot get any descriptors. 

import cv2

img2 = cv2.imread('images/grains/grains.jpg', 0)

# Initiate FAST object with default values
detector = cv2.FastFeatureDetector_create(50)   #Detects 50 points

kp = detector.detect(img, None)

img2 = cv2.drawKeypoints(img, kp, None, flags=0)

# cv2.imshow('Corners',img2)
# cv2.waitKey(0)


#############################################
#BRIEF (Binary Robust Independent Elementary Features)
#One important point is that BRIEF is a feature descriptor, 
#it doesn’t provide any method to find the features.
# Not going to show the example as BRIEF also not working in opencv 3

###############################################
#ORB
# Oriented FAST and Rotated BRIEF
# An efficient alternative to SIFT or SURF
# ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor

# import numpy as np
# import cv2

img3 = cv2.imread('images/grains/grains.jpg', 0)

orb = cv2.ORB_create(100)


kp, des = orb.detectAndCompute(img, None)
#todo: circle showa the strength of the keypoint and the line shows the angle of direction. 


# draw only keypoints location,not size and orientation
#img2 = cv2.drawKeypoints(img, kp, None, flags=None)
# Now, let us draw with rich key points, reflecting descriptors. 
# Descriptors here show both the scale and the orientation of the keypoint.
img2 = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("With keypoints", img2)
cv2.waitKey(0)