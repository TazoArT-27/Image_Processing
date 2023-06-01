
#*If the image cannot be segmented by histogram segmentation, then Random Walker Segmentation is used. 
#?WHAT IS RANDOM WALKER SEGMENTATION?
#?Answer:  It is an iterative algorithm that assigns labels to pixels in an image based on a user-defined set of markers or seeds.
#?The algorithm is inspired by the concept of a random walker moving on a graph. 
#?In the context of image segmentation, the graph represents the image, and each pixel in the image is considered as a node in the graph. 
#?The algorithm uses the intensity or color similarities between neighboring pixels to determine the probability of 
#?a random walker crossing from one pixel to another.

"""
Using Random walker to generate lables and then segment and finally cleanup using closing operation.
"""


import matplotlib.pyplot as plt
from skimage import io, img_as_float
import numpy as np


img = img_as_float(io.imread("Images/randomWalker/Alloy_noisy.jpg"))


# plt.hist(img.flat, bins=100, range=(0, 1)) 
# plt.show()

#*Very noisy image so histogram looks horrible. Let us denoise and see if it helps.

from skimage.restoration import denoise_nl_means, estimate_sigma

sigma_est = np.mean(estimate_sigma(img, channel_axis=2))
denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True, 
                               patch_size=5, patch_distance=3)
                           
# plt.hist(denoise_img.flat, bins=100, range=(0, 1)) 
# plt.show()

#todo Much better histogram and now we can see two separate peaks. 
#todo Still close enough so cannot use histogram based segmentation.
#todo Let us see if we can get any better by some preprocessing.
#todo Let's try histogram equalization

from skimage import exposure   #Contains functions for hist. equalization

#eq_img = exposure.equalize_hist(denoise_img)
eq_img = exposure.equalize_adapthist(denoise_img)
# plt.imshow(eq_img, cmap='gray')
# plt.hist(eq_img.flat, bins=100, range=(0., 1))
# plt.show()

#*Not any better. Let us stretch the hoistogram between 0.7 and 0.95

#*The range of the binary image spans over (0, 1).
#*For markers, let us include all between each peak.
markers = np.zeros(img.shape, dtype=np.uint)

markers[(eq_img < 0.8) & (eq_img > 0.7)] = 1
markers[(eq_img > 0.85) & (eq_img < 0.99)] = 2
# plt.imshow(markers)
# plt.show()
# plt.imsave("Images/randomWalker/markers.jpg", markers)

from skimage.segmentation import random_walker
# Run random walker algorithm
#* https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.random_walker
labels = random_walker(eq_img, markers, beta=10, mode='bf')
# plt.imshow(labels)
# plt.show()

segm1 = (labels == 1)
segm2 = (labels == 2)

all_segments = np.zeros((eq_img.shape[0], eq_img.shape[1], 3)) #nothing but denoise img size but blank
#!Blank image is created for making the segmented parts to show in that one blank image. 

all_segments[segm1] = (1,0,0)
all_segments[segm2] = (0,1,0)

# plt.imshow(all_segments)
# plt.show()

from scipy import ndimage as nd

segm1_closed = nd.binary_closing(segm1, np.ones((3,3)))
segm2_closed = nd.binary_closing(segm2, np.ones((3,3)))

all_segments_cleaned = np.zeros((eq_img.shape[0], eq_img.shape[1], 3)) 

all_segments_cleaned[segm1_closed] = (1,0,0)
all_segments_cleaned[segm2_closed] = (0,1,0)

plt.imshow(all_segments_cleaned) 
# plt.show()
# plt.imsave("Images/randomWalker/random_walker.jpg", all_segments_cleaned)

