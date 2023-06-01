
##*This lecture is on HISTOGRAM BASED IMAGE SEGMENTATION. 
##*This technique is going to work only if you can identify or distinguish different regions based on histograms only 
##*or based on gray level only. But if you have images with different texture but the gray levels are pretty much the same
##*then you need to do some texture based identification like "ENTORY FILTER" at first then this.

##!WHAT IS HISTOGRAM?
##?Answer: A histogram is a graphical representation or plot that shows the distribution of pixel intensities or colors in an image. 
# ?It provides a visual summary of the frequency of occurrence of different intensity or color values in the image.

##!WHAT IS HISTOGRAM BASED SEGMENTATION?
##?Answer: Histogram-based segmentation is a technique used to separate an image into distinct regions 
##?or objects based on the analysis of its histogram. The histogram of an image represents the distribution of pixel intensities 
##?or colors present in the image.The basic idea behind histogram-based segmentation is that different objects 
##?or regions in an image tend to have different pixel intensity or color distributions. 
##?By analyzing the histogram, it is possible to identify intensity/color ranges corresponding to different objects or regions.

# Histogram based segmentation

from skimage import io
from matplotlib import pyplot as plt
import numpy as np

img = io.imread("Images/histSeg/BSE_Google_noisy.jpg")
#plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')  

#Let's clean the noise using edge preserving filter.
#As mentioned in previous tutorial, my favorite is NLM

from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float
##*img_as_ubyte is used for converting the image into 8 bit again because we have converted the image into float.

float_img = img_as_float(img)
sigma_est = np.mean(estimate_sigma(float_img, channel_axis=2))


denoise_img = denoise_nl_means(float_img, h=1.15 * sigma_est, fast_mode=True, 
                               patch_size=5, patch_distance=3)
                           
denoise_img_as_8byte = img_as_ubyte(denoise_img)
# plt.imshow(denoise_img_as_8byte, cmap=plt.cm.gray, interpolation='nearest')
# plt.show()

#Let's look at the histogram to see how many peaks we have. 
#Then pick the regions for our histogram segmentation.

# plt.hist(denoise_img_as_8byte.flat, bins=100, range=(0,255))  #.flat returns the flattened numpy array (1D)
# plt.hist(denoise_img_as_8byte.flat, bins=100, range=(0,100)) 
# plt.show()


segm1 = (denoise_img_as_8byte <= 57)
segm2 = (denoise_img_as_8byte > 57) & (denoise_img_as_8byte <= 110)
segm3 = (denoise_img_as_8byte > 110) & (denoise_img_as_8byte <= 210)
segm4 = (denoise_img_as_8byte > 210)

#?How to show all these images in single visualization?
#*Construct a new empty image with same shape as original except with 3 layers.
#print(median_img.shape[0])
all_segments = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3)) #nothing but denoise img size but blank
"""
todo Here's a breakdown of the code:

    todo 1. denoise_img_as_8byte is assumed to be an image that has undergone denoising or preprocessing, and its shape is (height, width).

    todo 2. denoise_img_as_8byte.shape[0] gives the height of the image (number of rows), and 
    todo    denoise_img_as_8byte.shape[1] gives the width of the image (number of columns).

    todo 3. The line all_segments = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3)) 
    todo    initializes a 3-dimensional numpy array filled with zeros. The dimensions of the array are (height, width, 3), 
    todo    where the third dimension represents the color channels (red, green, and blue) of the image.
"""

all_segments[segm1] = (1,0,0)
all_segments[segm2] = (0,1,0)
all_segments[segm3] = (0,0,1)
all_segments[segm4] = (1,1,0)
# plt.imshow(all_segments)
# plt.show()
plt.imsave("Images/histSeg/segmented_before_cleaning.jpg", all_segments);

#Lot of yellow dots, red dots and stray dots. how to clean
#We can use binary opening and closing operations. Open takes care of isolated pixels within the window
#Closing takes care of isolated holes within the defined window

from scipy import ndimage as nd

segm1_opened = nd.binary_opening(segm1, np.ones((5,5)))
segm1_closed = nd.binary_closing(segm1_opened, np.ones((5,5)))

segm2_opened = nd.binary_opening(segm2, np.ones((5,5)))
segm2_closed = nd.binary_closing(segm2_opened, np.ones((5,5)))

segm3_opened = nd.binary_opening(segm3, np.ones((5,5)))
segm3_closed = nd.binary_closing(segm3_opened, np.ones((5,5)))

segm4_opened = nd.binary_opening(segm4, np.ones((5,5)))
segm4_closed = nd.binary_closing(segm4_opened, np.ones((5,5)))

all_segments_cleaned = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3)) #nothing but 714, 901, 3

all_segments_cleaned[segm1_closed] = (1,0,0)
all_segments_cleaned[segm2_closed] = (0,1,0)
all_segments_cleaned[segm3_closed] = (0,0,1)
all_segments_cleaned[segm4_closed] = (1,1,0)

# plt.imshow(all_segments_cleaned)  #All the noise should be cleaned now
# plt.show()
plt.imsave("Images/histSeg/segmented_after_cleaning.jpg", all_segments_cleaned);