from skimage import io 
from matplotlib import pyplot as plt
import numpy as np

img1 = io.imread("Images/ad.jpg", as_gray=True);

from skimage.transform import resize, rescale, downscale_local_mean
rescaled_image = rescale(img1, 2.0/4.0, anti_aliasing=False);
resized_image = resize(img1, (200, 200));
down_scaled_image = downscale_local_mean(img1, (4,3)); 

# plt.imshow(rescaled_image);
# plt.imshow(resized_image);
# plt.imshow(down_scaled_image);

from skimage.filters import roberts, sobel, scharr, prewitt
# roberts_img = roberts(img1);
# plt.subplot(4,2,1);
# plt.imshow(roberts(img1));
# plt.subplot(4,2,2);
# plt.imshow(sobel(img1));
# plt.subplot(4,2,3);
# plt.imshow(scharr(img1));
# plt.subplot(4,2,4);
# plt.imshow(prewitt(img1));

from skimage.feature import canny
# edge_canny = canny(img1, sigma=1);
# plt.imshow(edge_canny);

from skimage import data, restoration
psf = np.ones((3, 3)) / 9;
# Perform Richardson-Lucy deconvolution
deconvolved_image = restoration.richardson_lucy(img1, psf, num_iter=1000);
plt.imshow(deconvolved_image, cmap='gray');

plt.show();

"""
!rescaling:
*In scikit-image (skimage), the rescale function is used to adjust the scale or range of pixel values in an image. 
*It is primarily used for contrast enhancement and normalization of images.
*The rescale function takes an input image and rescales its pixel values to a specified range. By default, 
*the function scales the image to the range [0, 1], where the minimum pixel value becomes 0 and the maximum pixel value becomes 1. 
*This is often referred to as normalization. This can be useful for various tasks, 
*such as preparing images for machine learning algorithms that require inputs in a specific range.

!anti-aliasing:
*The anti_aliasing parameter determines whether anti-aliasing should be applied during the rescaling process. 
*Anti-aliasing is a technique used to reduce aliasing artifacts, such as jagged edges, that can occur when reducing the size of an image. 
*Setting anti_aliasing=False means that no anti-aliasing will be performed, 
*and the resulting image may have aliasing artifacts if the scaling factor is significant.

!pixel reducation:
*In the code above, you need to replace 'image.jpg' with the path to the actual image file you want to rescale. 
*The resulting rescaled_image will have dimensions that are 2.0/4.0 times the original image's dimensions.

!resizing:
*this means the image is resized to 200x200 pixels

!downscale_local_image:
*The resulting down_scaled_image will have dimensions that are reduced by a factor of 4 in the horizontal direction and 3 in the vertical direction.
*The downscale_local_mean function divides the image into "non-overlapping blocks" defined by the block size, and within each block, 
*it computes the mean value of the pixels. The resulting down-scaled image has fewer pixels, 
*and each pixel represents the mean value of the corresponding block in the original image.
*By adjusting the block size, you can control the level of downsampling and the amount of detail retained in the down-scaled image. 
?Smaller block sizes preserve more details but result in a higher-resolution down-scaled image, 
?while larger block sizes reduce more details but result in a lower-resolution down-scaled image.
*It's important to note that the downscale_local_mean function performs a type of average pooling and may not 
*preserve all the fine details and sharpness of the original image. 
*The choice of downsampling method depends on the specific application and the desired trade-off between reduced size and 
*retained image information.

?WHAT IS DECONVOLUTION IN SKIMAGE?
*In scikit-image (skimage), the deconvolution module provides functions for performing image deconvolution, 
*which is a process of estimating the original image from its degraded version. 
*Image deconvolution is often used to improve image quality by reducing blurring or artifacts caused by factors like motion blur or 
*lens imperfections.
*The deconvolution module in skimage.feature includes various algorithms for deconvolution, 
*such as Richardson-Lucy deconvolution, Wiener deconvolution, and total variation deconvolution.

?WHAT IS POINT SPREAD FUNCTION(PSF)?
*The Point Spread Function (PSF) is a key component in image deconvolution algorithms. 
*It represents the blurring effect or the degradation that occurs during the image formation process, 
*such as through optical systems or motion blur. 
*The PSF describes how a point source in the original image spreads out and affects neighboring pixels in the degraded image.

?WHAT IS GAUSSIAN KERNEL SKIMAGE?
*When a Gaussian kernel is applied to an image using convolution, 
*each pixel in the image is weighted by the corresponding value in the kernel matrix and the neighboring pixels. 
*This weighted averaging process results in a smoothing or blurring effect, 
*where the intensity values of neighboring pixels contribute to the final value of each pixel.
*The Gaussian kernel is widely used in image processing tasks, such as noise reduction, image smoothing, and feature detection. 
*It is favored for its ability to reduce noise and blur while preserving important image details and edges. 
*The choice of the kernel size and standard deviation depends on the specific application and the desired trade-off 
*between noise reduction and preservation of fine details.
todo GAUSSIAN KERNEL CAN BE USED AS A POINT SPREAD FUNCTION

?Why ENTROPY is used for?
*For restoration purpose in an image. 



"""