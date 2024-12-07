#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:\Users\jackb\OneDrive\Documents\server_new\img_1.jpg",
                 cv2.IMREAD_GRAYSCALE)  # rest of what?
lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
lap = np.uint8(np.absolute(lap))
sobelX = cv2.Sobel(img, -1, 1, 0)
sobelY = cv2.Sobel(img, -1, 0, 1)
edges = cv2.Canny(img, 200, 200)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
gradients_sobelxy = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)
sobelCombined = cv2.bitwise_or(sobelX, sobelY)
cv2.imwrite('C:\Users\jackb\OneDrive\Documents\server_new\saved_imgs', edges)
titles = ['image', 'Laplacian', 'sobelX', 'sobelY', 'sobelCombined', 'Canny']
images = [img, lap, sobelX, sobelY, sobelCombined, edges]
for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()


# In[14]:


image = cv2.imread("C:\Users\jackb\OneDrive\Documents\server_new\img_1.jpg")

new_image = image.copy()

# lfggggg. what are the numbers now?
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display grayscale
ret, binary = cv2.threshold(gray, 100, 255,
                            cv2.THRESH_OTSU)


# invert colours
inverted_binary = ~binary

# Find contours and store in list
contours, hierarchy = cv2.findContours(inverted_binary,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

# Contours to red color
with_contours = cv2.drawContours(image, contours, -1, (255, 0, 255), 3)

# total no of contours detected
print('Total number of contours detected: ' + str(len(contours)))


# bounding box around the third contour
x, y, w, h = cv2.boundingRect(contours[len(contours)-1])
cv2.rectangle(with_contours, (x, y), (x+w, y+h), (255, 0, 0), 5)
cv2.imshow('Third contour with bounding box', with_contours)
x_coord = []
y_coord = []
width = []
height = []
a = 0
ROI_number = 0
prevx = 0
# Draw a bounding box around all contours and crop it out into seperate image files
for c in contours:
    if (cv2.contourArea(c)) > 1000:
        cv2.rectangle(with_contours, (x, y), (x+w, y+h), (255, 0, 0), 5)
        x, y, w, h = cv2.boundingRect(c)
        print(x, y, w, h)
        x_coord.append(x)
        y_coord.append(y)
        width.append(w)
        height.append(h)
        a = x_coord[-1]
        if abs(a-prevx) <= 9:
            x_coord.pop(-1)
            y_coord.pop(-1)
            height.pop(-1)
            width.pop(-1)
        prevx = a


print("x coord is  ", x_coord)
print("y coord is ", y_coord)
print("height is ", height)
print("width is ", width)

# okay so now its a matter of sending these numbers over to blender, using the sever.py file?
# so maybe i can copy this code intoo that file, and then change the vars?
# okay lets see
image = cv2.imread("C:\Users\jackb\OneDrive\Documents\server_new\img_1.jpg")
for i in range(0, len(x_coord)):
    # print(y_coord[i], height[i], x_coord[i], width[i])
    ROI = image[y_coord[i]:y_coord[i]+height[i],
                x_coord[i]:x_coord[i]+width[i]]
    cv2.imwrite('C:\Users\jackb\OneDrive\Documents\server_new\saved_imgs\ROI_{}.png'.format(ROI_number), ROI)
    # cv2.rectangle(new_image,(x,y),(x+w,y+h),(36,255,12),2)
    ROI_number += 1

cv2.imshow('All contours with bounding box', with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:
