import cv2
import numpy as np
import os
import pandas as pd
import time
start_time=time.time()
test_original=[]
test_original = cv2.imread("right_eye.bmp")
cv2.imshow("Original", cv2.resize(test_original, None, fx=1, fy=1))
cv2.waitKey(0)
cv2.destroyAllWindows()
count=0
x=False
for file in [file for file in os.listdir("final")]:
    iris_database_image = cv2.imread("./final/" + file)

    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(test_original, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(iris_database_image, None)

    matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10),
                                    dict()).knnMatch(descriptors_1, descriptors_2, k=2)
    match_points = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)
        keypoints = 0
        if len(keypoints_1) <= len(keypoints_2):
            keypoints = len(keypoints_1)
        else:
            keypoints = len(keypoints_2)
        if (len(match_points) / keypoints) > 0.95:
            print("The input Iris image is matched!!")
            print("percentage(%) of match: ", len(match_points) / keypoints * 100)
            b=time.time()-start_time
            print("the total time taken is :{} Seconds".format(b))
            result = cv2.drawMatches(test_original, keypoints_1, iris_database_image,
                                    keypoints_2, match_points, None)
            result = cv2.resize(result, None, fx=1, fy=1)
            cv2.imshow("result", result)
            cv2.waitKey(10000)
            cv2.destroyAllWindows()
            x=True
            count+=1
            break
    if x:
        break
else:
    print("The input Iris image is not matched!!")