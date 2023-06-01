
##*Practice of 3 types of digital filters for denoising purpose. 
##*digital filters are mixture of kernel and an image



# Filters work by convolution with a moving window called a kernel.
#Convolution is nothing but multiplication of two arrays of different sizes. 
#The image will be of one size and the kernel with be of a different size, 
# #usually much smaller than image
# The input pixel is at the centre of the kernel. 
# The convolution is performed by sliding the kernel over the image, 
# $usually from top left of image.
# Linear filters and non-linear filters.
# Gaussian is an example of linear filter. 
#Non-linear filters preserve edges. 
#Median filter is an example of non-linear filter. 
#The algorithm selects the median value of all the pixels in the selected window
#NLM: https://scikit-image.org/docs/dev/auto_examples/filters/plot_nonlocal_means.html

##***************Gaussian Kernel******************
##? What is Gaussian Kernel ? => This is basically a matrix, which is a result of two other matrix's multiplication. 
##todo: For better understanding Gaussian Kernel go through this video: https://www.youtube.com/watch?v=EhUFckuZhUQ . 
import numpy 
from matplotlib import pyplot as plt

# def gaussian_kernel(size, size_y=None):
#     size = int(size)
#     if not size_y:
#         size_y = size
#     else:
#         size_y = int(size_y)
#     x, y = numpy.mgrid[-size:size+1, -size_y:size_y+1]
#     g = numpy.exp(-(x**2/float(size)+y**2/float(size_y)))
#     return g / g.sum()
 

# gaussian_kernel_array = gaussian_kernel(3)
# print(gaussian_kernel_array)
# plt.imshow(gaussian_kernel_array, cmap=plt.get_cmap('jet'), interpolation='nearest')
# plt.colorbar()
# plt.show()

############################ Denoising filters ###############
##! Disadvantages: Gaussian filter cannot preserve the edges and blurs the image that's why it becomes difficult to extract information.
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float
from matplotlib import pyplot as plt
from skimage import io
import numpy as np

##*It is important to change the image into float if we are doing any math on the image.
img = img_as_float(io.imread("Images/denoising/noisy_img.jpg"))

from scipy import ndimage as nd
gaussian_img = nd.gaussian_filter(img, sigma=3)
##?WHAT DOES SIGMA DO? => If the value of sigma is high then image will be smooth but blured,
##?On the other hand if the value of sigma is low then it is less smoother and less blured.
plt.imsave("Images/denoising/gaussian.jpg", gaussian_img)

##! Disadvantages: Median filter also blures many important information.
median_img = nd.median_filter(img, size=3)
plt.imsave("Images/denoising/median.jpg", median_img)

# gaussian_img = nd.gaussian_filter(img, sigma=3)
# plt.imsave("Images/gaussian.jpg", gaussian_img)


##### NLM#####
#todo Non Local Means is the best filter for working because it doesn't affects the texture and edges of the image. 
sigma_est = np.mean(estimate_sigma(img, channel_axis=2))
##*channel_axis=2 means it is a color image, if 1 then it is grayscale image. 

patch_kw = dict(patch_size=5,      
                patch_distance=3,  
                multichannel=True)

denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True,
                               patch_size=5, patch_distance=3)
"""
denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
"""
denoise_img_as_8byte = img_as_ubyte(denoise_img)

plt.imshow(denoise_img)
#plt.imshow(denoise_img_as_8byte, cmap=plt.cm.gray, interpolation='nearest')
plt.imsave("Images/denoising/NLM.jpg",denoise_img)