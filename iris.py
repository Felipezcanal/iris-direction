import argparse
import cv2
from pprint import pprint
import numpy as np
import pandas as pd
g_minDist = 30
g_dp = 1.1109*10000
def parameter1(minDist):
	global g_minDist
	g_minDist = minDist
	detectCircles(cimg, image, minDist, g_dp)

def parameter2(y):
	global g_dp
	g_dp = dp
	detectCircles(cimg, image, g_minDist, dp)

def detectCircles(image ,colored_image, minDist, dp):
	# pprint(minDist)
	# pprint(dp)
	i_colored_images = colored_image.copy()
	i_image = image.copy()
	blur = cv2.medianBlur(i_image,5)
	circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp/10000, minDist)
	circles = np.uint16(np.around(circles))
	for count, i in enumerate(circles[0,:]):

		cv2.circle(i_colored_images,(i[0],i[1]),i[2],(0,255,0),2) #circle
		cv2.circle(i_colored_images,(i[0],i[1]),1,(0,0,255),2) # center point
		cv2.putText(i_colored_images, str(count+1), (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
	cv2.imshow('coins',i_colored_images)

def handleVideo():


	video_capture = cv2.VideoCapture("IMG_1460.MOV")



	cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)
	cv2.namedWindow('gray', cv2.WINDOW_KEEPRATIO)

	# while True:
	if not video_capture.isOpened():
		print('Unable to load camera.')
		sleep(5)
		pass
	# Capture frame-by-frame
	ret, frame = video_capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	detectCircles(gray, frame, g_minDist, g_dp)


# image = cv2.imread("coins2.jpg", cv2.IMREAD_COLOR)
# cimg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.namedWindow("coins", cv2.WINDOW_KEEPRATIO)
#
cv2.createTrackbar('Parameter1','coins',5000,15000,parameter2)
cv2.createTrackbar('Parameter2','coins',0,100,parameter1)
# print('teste')
while True:
	# cv2.imshow("coins", image.astype('uint8'))
	key = cv2.waitKey(1) & 0xFF
	if key == ord("d"):
		handleVideo()

		# detectCircles(cimg, image, g_minDist, g_dp)
	elif key == ord("c"):
		break
cv2.destroyAllWindows()
