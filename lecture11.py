import numpy as np;
import imageio.v2 as imageio;   ##*used for image reading and showing in matrix formate 

img = imageio.imread("Images/2dc.jpg");
#print(img);
#print(img.shape);
tint_color = np.array([0., 0., 1.]);
img = np.expand_dims(img, axis=2)
img_tinted = img * tint_color;
img_tinted = img_tinted.astype(np.uint8);
imageio.imwrite('Images/new.jpg', img_tinted);


"""
* features are of two types. 
* 1. numerical type & 2. categorical type
* categorical type features are basically object type
"""


"""
*normal comment
!alert comment
?question comment
todo: todo comment
@param parameter comment
"""
##*highlight comment
##!alert comment
##?question comment
##todo todo comment