import cv2
import numpy as np
import argparse
import imutils
img = cv2.imread('circle.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


edges = cv2.Canny(gray,100,200,apertureSize = 3)
cv2.waitKey(0)

minLineLength = 30
maxLineGap = 20
lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength=minLineLength,maxLineGap=maxLineGap)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.waitKey(0)
"""
final_circles = [] 
circles = cv2.HoughCircles(img,method = CV_HOUGH_GRADIENT, dp=1,minDist = 20,param1=50,param2=30,minRadius=0,maxRadius=0)
rows = img.shape[0] 
cols = img.shape[1]
circles = np.round(circles[0, :]).astype("int") 
for (x, y, r) in circles: 
    if (r <= x <= cols-1-r) and (r <= y <= rows-1-r): 
        final_circles.append([x, y, r])

final_circles = np.asarray(final_circles).astype("int")
"""
class ShapeDetector:

	def __init__(self):
		pass

	def detect(self, c):
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)


		if len(approx) == 3:
			shape = "triangle"
		
		elif len(approx) == 4:

			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)
			shape = "square" if ar >= 0.95 and ar <= 1.15 else "rectangle"

		else:
			shape = "circle"
		return shape



image = img
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()


for c in cnts:
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)

	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (100, 100, 100), 2)
	cv2_imshow(image)
	cv2.waitKey(0)