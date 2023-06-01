import cv2

from skimage import filters



img = cv2.imread("Images/ad.jpg",0)
img2 = filters.sobel(img);
cv2.imshow("pic",img);
cv2.imshow("pic2", img2);
print(img.shape);
cv2.waitKey(0);
cv2.destroyAllWindows();
