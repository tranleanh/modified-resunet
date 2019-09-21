import cv2
import numpy as np
import glob
import os

def nothing(*arg):
	pass

if __name__ == '__main__':

	icol = (28, 0, 0, 56, 255, 255)  # Green

	cv2.namedWindow('colorTest')
	# Lower range colour sliders.
	cv2.createTrackbar('lowHue', 'colorTest', icol[0], 255, nothing)
	cv2.createTrackbar('lowSat', 'colorTest', icol[1], 255, nothing)
	cv2.createTrackbar('lowVal', 'colorTest', icol[2], 255, nothing)
	# Higher range colour sliders.
	cv2.createTrackbar('highHue', 'colorTest', icol[3], 255, nothing)
	cv2.createTrackbar('highSat', 'colorTest', icol[4], 255, nothing)
	cv2.createTrackbar('highVal', 'colorTest', icol[5], 255, nothing)

	# img_name = "test.png"

	srcImageName = glob.glob("./labelled_input/*.png")

	for imgName in srcImageName:

		lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
		lowSat = cv2.getTrackbarPos('lowSat', 'colorTest')
		lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
		highHue = cv2.getTrackbarPos('highHue', 'colorTest')
		highSat = cv2.getTrackbarPos('highSat', 'colorTest')
		highVal = cv2.getTrackbarPos('highVal', 'colorTest')

		frame = cv2.imread(imgName)
		name = os.path.basename(imgName)
		print(name)

		frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		colorLow = np.array([lowHue, lowSat, lowVal])
		colorHigh = np.array([highHue, highSat, highVal])
		mask = cv2.inRange(frameHSV, colorLow, colorHigh)
		


		# Show the mask and the image
		cv2.imshow('Threshoding', mask)
		cv2.imshow('Input', frame)
		
		# Write label
		cv2.imwrite("./bw_label/" + name, mask)

		k = cv2.waitKey(5) & 0XFF
		if k == 27:
			break

	cv2.destroyAllWindows()