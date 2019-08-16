import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('guava.jpg')
greyimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.subplot(2,2,1)
plt.imshow(greyimage,cmap="gray")
laplacian = cv2.Laplacian(greyimage,cv2.CV_64F)
plt.subplot(2,2,2)
plt.imshow(laplacian,cmap="gray")
plt.show()
