import glob
import cv2
img=glob.glob("final/*.bmp")
cv2.imshow(img,0)
cv2.waitKey(0)
cv2.DestroyallWindows()