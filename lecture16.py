from skimage import io 
import numpy as np 
from matplotlib import pyplot as plt 
import cv2
import imageio

img1 = io.imread("Images/main.avif");
#print(img1);
# print(img1.min(), img1.max());

# float_img1 = img1.astype(np.float32) / 255.0;
# print(float_img1.min(), float_img1.max());


# fig1, ax1 = plt.subplots()
# ax1.imshow(img1)
# ax1.set_title('Figure 1')

# Create Figure 2 and its plot
# fig2, ax2 = plt.subplots()
# ax2.imshow(float_img1)
# ax2.set_title('Figure 2')

# Show both figures
# plt.show()

img1[10:200, 12:200, :] = [255,255,0];
plt.imshow(img1);
plt.show();


# random_image = np.random.random([50, 50]);
# plt.imshow(random_image);
# plt.show();
# io.imsave("Images/random.jpg", random_image);

#print(random_image);
# print(random_image.min(), random_image.max());
