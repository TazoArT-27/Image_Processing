from scipy import misc, ndimage
import imageio

def rotated(img):
    rotated_image = ndimage.rotate(img,45);
    imageio.imwrite('Images/rotated.jpg', rotated_image);
    
def blurred(img):
    blurred_image = ndimage.gaussian_filter(img, sigma=5);
    imageio.imwrite('Images/blurred.jpg', blurred_image);

def denoised(img):
    denoised_image = ndimage.median_filter(img, 3);
    imageio.imwrite('Images/deniosed.jpg', denoised_image);
    
img1 = imageio.imread('Images/ad.jpg');

rotated(img1);
blurred(img1);
denoised(img1);