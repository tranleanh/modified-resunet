import cv2
import glob
import os

srcImageName = glob.glob("./gen_input/*.jpg")

for imgName in srcImageName:
	img = cv2.imread(imgName)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	name = os.path.basename(imgName)
	print(name)

	cv2.imwrite(name, img)

	# Convert to binary
	# ret, bw_img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

	# name = os.path.basename(imgName)
	# print(name)

	# cv2.imshow("Binary Image", bw_img)
	# cv2.imwrite(name, bw_img)

	if cv2.waitKey(20) == 27:
		break

# cv2.destroyAllWindows()