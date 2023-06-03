
#*Image registration using homography in openCV
#?WHAT IS ORB IN OPENCV?
#*ORB (Oriented FAST and Rotated BRIEF) is a keypoint detector and descriptor algorithm available in OpenCV. 
#*It combines the FAST (Features from Accelerated Segment Test) keypoint detector and the BRIEF 
#*(Binary Robust Independent Elementary Features) descriptor.

#?WHAT IS HOMOGRAPHY IN COMPUTER VISION?
#*In computer vision, homography refers to a 2D projective transformation that maps points from one plane to another. 
#*It is a transformation that preserves straight lines and allows for geometric distortion, rotation, translation, and 
#*scaling between two images or scenes.
#*A homography matrix (also known as a transformation matrix or a homography matrix) is a 3x3 matrix that represents 
#*the relationship between the points in two images or scenes. It defines the mapping between the coordinate systems of the two planes.

#?WHAT IS RANSAC IN HOMOGRAPHY IN COMPUTER VISION?
#*RANSAC (Random Sample Consensus) is a robust estimation algorithm commonly used in computer vision for s
#*olving problems like estimating the homography matrix between two sets of correspondences with outliers or noise. 
#*It is particularly useful when there is a possibility of incorrect or mismatched correspondences.



#TODO: Steps for the ORB: 
#todo                       1. import 2 images, 
#todo                       2. convert to gray scale, 
#todo                       3. Initiate ORB detector, 
#todo                       4. Find keypoints and describe them, 
#todo                       5. Match the keypoints by brute force matcher
#todo                       6. RANSAC (reject bad keypoints)
#todo                       7. Register 2 images (use homology)

import numpy as np
import cv2
from matplotlib import pyplot as plt

im1 = cv2.imread('images/monkey_distorted.jpg')          # Image that needs to be registered.
im2 = cv2.imread('images/monkey.jpg') # trainImage

img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Initiate ORB detector
orb = cv2.ORB_create(50)  #Registration works with at least 50 points

# find the keypoints and descriptors with orb
kp1, des1 = orb.detectAndCompute(img1, None)  #kp1 --> list of keypoints
kp2, des2 = orb.detectAndCompute(img2, None)

#Brute-Force matcher takes the descriptor of one feature in first set and is 
#matched with all other features in second set using some distance calculation.
# create Matcher object

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
#*matcher takes one keypoint of distorted image and tries to match it to all of the keypoints of the distorted image. 

# Match descriptors.
matches = matcher.match(des1, des2, None)  #Creates a list of all matches, just like keypoints
#*matches mainly calculates some type of distance between keypoints of desc1 and desc2. 

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

#Like we used cv2.drawKeypoints() to draw keypoints, 
#cv2.drawMatches() helps us to draw the matches. 
#https://docs.opencv.org/3.0-beta/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
# Draw first 10 matches.
img3 = cv2.drawMatches(im1,kp1, im2, kp2, matches[:10], None)

cv2.imshow("Matches image", img3)
cv2.waitKey(0)

#Now let us use these key points to register two images. 
#Can be used for distortion correction or alignment
#For this task we will use homography. 
# https://docs.opencv.org/3.4.1/d9/dab/tutorial_homography.html

# Extract location of good matches.
# For this we will use RANSAC.
#RANSAC is abbreviation of RANdom SAmple Consensus, 
#in summary it can be considered as outlier rejection method for keypoints.
#http://eric-yuan.me/ransac/
#RANSAC needs all key points indexed, first set indexed to queryIdx
#Second set to #trainIdx. 

points1 = np.zeros((len(matches), 2), dtype=np.float32)  #Prints empty array of size equal to (matches, 2)
points2 = np.zeros((len(matches), 2), dtype=np.float32)
#*This is nothing but initializing arrays with zeroes and zeroes will be later replaced by the keypoints. 

for i, match in enumerate(matches):
   points1[i, :] = kp1[match.queryIdx].pt    #gives index of the descriptor in the list of query descriptors
   points2[i, :] = kp2[match.trainIdx].pt    #gives index of the descriptor in the list of train descriptors

#Now we have all good keypoints so we are ready for homography.   
# Find homography
#https://en.wikipedia.org/wiki/Homography_(computer_vision)
  
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
#*RANSAC is like preprocessing before homography. 
 
  # Use homography
height, width, channels = im2.shape
im1Reg = cv2.warpPerspective(im1, h, (width, height))  #Applies a perspective transformation to an image.
   
print("Estimated homography : \n",  h)

cv2.imshow("Registered image", im1Reg)
cv2.waitKey()