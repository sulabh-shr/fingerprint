import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift

path = r"D:\workspace\datasets\MMFV-Dataset\2\1\f1\Pitch\1_1.jpg"

img = cv2.imread(path)
img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret1, th1 = cv2.threshold(img_grey, 127, 255, cv2.THRESH_OTSU)
plt.imshow(th1, cmap='grey')
plt.show()
plt.close()
