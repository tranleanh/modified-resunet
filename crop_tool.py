from matplotlib import pyplot as plt
import cv2
import time

img = cv2.imread('po_7341_pan_0000000.jpg')

numCols = 7
numRows = 7

pieces = []

for i in range(numRows):
	for j in range(numCols):
		from_x = j*img.shape[0]//numCols
		from_y = i*img.shape[1]//numRows
		to_x = (j+1)*img.shape[0]//numCols
		to_y = (i+1)*img.shape[1]//numRows
		cropped = img[from_y:to_y, from_x:to_x]

		img_name = str(i)+str(j)+'_7341_7x7.jpg'

		cv2.imwrite(img_name, cropped)

		print(img_name)

		# pieces.append(cropped)

# fig = plt.figure(figsize=(12, 10))
# for i in range(len(pieces)):
#     fig.add_subplot(numCols, numRows, i+1)
#     plt.imshow(pieces[i])

# plt.show()