import cv2
import glob
import os

srcImgName = glob.glob("./input/*.jpg")
trainImgName = glob.glob("./bw_label/*.png")

for i in range(len(srcImgName)):
    
    _srcImg = cv2.imread(srcImgName[i])
    name_srcImg = os.path.basename(srcImgName[i])
    name_trainImg = os.path.basename(trainImgName[i])

    flipVertical = cv2.flip(_srcImg, 0)
    flipHorizontal = cv2.flip(_srcImg, 1)
    flipBoth = cv2.flip(_srcImg, -1)

    _trainImg = cv2.imread(trainImgName[i])
    # name_trainImg = os.path.basename(imgName)

    flipVertical_mask = cv2.flip(_trainImg, 0)
    flipHorizontal_mask = cv2.flip(_trainImg, 1)
    flipBoth_mask = cv2.flip(_trainImg, -1)

    name_gen_0 = "0_" + name_srcImg   
    name_gen_1 = "1_" + name_srcImg   
    name_gen_2 = "2_" + name_srcImg  

    label_name_gen_0 = "0_" + name_trainImg   
    label_name_gen_1 = "1_" + name_trainImg   
    label_name_gen_2 = "2_" + name_trainImg  

    cv2.imwrite(name_gen_0, flipVertical)
    cv2.imwrite(name_gen_1, flipHorizontal)
    cv2.imwrite(name_gen_2, flipBoth)

    cv2.imwrite(label_name_gen_0, flipVertical_mask)
    cv2.imwrite(label_name_gen_1, flipHorizontal_mask)
    cv2.imwrite(label_name_gen_2, flipBoth_mask)