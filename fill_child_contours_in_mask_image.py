import cv2
import numpy as np
import os


def child_contours_filler(src_image_path):

    src_mask_image = cv2.imread(src_image_path)
    resultant_image = src_mask_image.copy()
    # src_mask_image = cv2.medianBlur(src_mask_image,3)
    imageGray = cv2.cvtColor(src_mask_image,cv2.COLOR_BGR2GRAY)
   
    # Find all contours in the image using RETE_CCOMP method 
    contours, hierarchy = cv2.findContours(imageGray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # Loop over all the contours detected
    for i,cont in enumerate(contours):
        
        # If the contour is at first level draw it in green 
        if hierarchy[0][i][3] == -1:
        
            # cv2.drawContours(src_mask_image, cont, -1, (0,255,0), 3)
            None
            
        # else draw the contour in Red
        else:
            cnt = np.array(cont, dtype=np.int32)
            cnt = cnt.reshape((-1, 1, 2))
            cv2.fillPoly(resultant_image,[cnt],(255,255,255))
            
    return resultant_image 

src_mask_image_path = "test_results_aug_2024/final_mask_image_1.png"

resultant_image = child_contours_filler(src_mask_image_path)
resultant_image = cv2.GaussianBlur(resultant_image,(5,5),0)
cv2.imwrite("child_contours_filled_mask_image.png",resultant_image)



