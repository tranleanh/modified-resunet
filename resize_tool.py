import cv2
import glob
import os

img = cv2.imread("./7x7/02_7341_7x7.png")
img = cv2.resize(img, (1572, 1572))

cv2.imshow("Resized Image", img)
cv2.imwrite('resized_1572.png', img)
cv2.waitKey()
cv2.destroyAllWindows()