"""
*For image processing basically 4 libraries are used. 
*These are: Pillow, Matplotlib, Scikit Image, OpenCV.

*For basic image handling pillow is used. But it not used for ML or Computer Vision. 
todo matplotlib is basically a plotting library.
!scikit image is a image processing library. Image senmentation. 
            *scikit image can do
                *image I/O
                *image transformation: Perform various transformations on images, such as resizing, 
                                       *cropping, rotating, flipping, and warping.
                *image segmentation
                *image filtering
                *image enhancement
                *image restoration
                *feature detection and extraction
                *image registration
                *morphological operation
                *image analysis

?openCV is mainly built for Computer Vision
"""

import cv2
from matplotlib import pyplot as plt 
import glob

# img1 = cv2.imread("Images/os.jpg", 0);
# img2 = cv2.imread("Images/os.jpg", 1);

# desired_width = 600;
# desired_height = 400;

# resized_image1 = cv2.resize(img1, (desired_width, desired_height));
# resized_image2 = cv2.resize(img2, (desired_width, desired_height));

# cv2.imshow("pic1",resized_image1);
# cv2.imshow("pic2",resized_image2);

path = "Images/*";
for file in glob.glob(path):
    imgs = cv2.imread(file);
    cv2.imshow("pic",imgs);
    cv2.waitKey(0);
    cv2.destroyAllWindows();
