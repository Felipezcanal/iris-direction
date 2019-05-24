import argparse
import cv2
from pprint import pprint
import numpy as np
import pandas as pd
g_minDist = 10
g_dp = 1.2889*10000
gray = 0
frame = 0
radiusmin = 27
radiusmax = 28
cannymin = 1
cannymax = 18

firsteye = (0,0)
secondeye = (0,0)

globalcount = 0;
def parameter1(minDist):
	global g_minDist
	g_minDist = minDist
	detectCircles(gray, frame, g_minDist, g_dp)

def parameter2(dp):
	global g_dp
	g_dp = dp
	detectCircles(gray, frame, g_minDist, g_dp)

def radiusminfunc(min):
	global radiusmin
	radiusmin = min
	detectCircles(gray, frame, g_minDist, g_dp)

def radiusmaxfunc(max):
	global radiusmax
	radiusmax = max
	detectCircles(gray, frame, g_minDist, g_dp)

def cannyminfunc(min):
	global cannymin
	cannymin = min
	detectCircles(gray, frame, g_minDist, g_dp)

def cannymaxfunc(max):
	global cannymax
	cannymax = max
	detectCircles(gray, frame, g_minDist, g_dp)

def detectCircles(image ,colored_image, minDist, dp):
	global radiusmax, radiusmin, cannymax, cannymin, firsteye, secondeye, globalcount
	# pprint(minDist)
	# pprint(dp)
	i_colored_images = colored_image.copy()
	i_image = image.copy()
	blur = cv2.medianBlur(i_image,5)
	# canny = cv2.Canny(blur, cannymin, cannymax)
	circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp/10000, minDist, param1=cannymin, param2=cannymax, minRadius=radiusmin, maxRadius=radiusmax)
	if circles is None:
		pprint('nada')
		return
	circles = np.uint16(np.around(circles))
	for count, i in enumerate(circles[0,:]):
		# pprint(i[0])
		if not ( ((250 <=i[0]<= 400) or (580 <=i[0]<= 700)) and (160 <=i[1]<= 240) ):
			continue

		if firsteye == (0,0) and (250 <=i[0]<= 400):
			firsteye = (i[0],i[1])
		elif secondeye == (0,0) and (580 <=i[0]<= 700):
			secondeye = (i[0],i[1])
		else:
			if (250 <=i[0]<= 400) and not(firsteye == (0,0)):
				cv2.arrowedLine(i_colored_images, firsteye, (i[0],i[1]), (0,0,255), 2)
			if (580 <=i[0]<= 700) and not(secondeye == (0,0)):
				cv2.arrowedLine(i_colored_images, secondeye, (i[0],i[1]), (0,0,255), 2)

		cv2.circle(i_colored_images,(i[0],i[1]),i[2],(0,255,0),2) #circle
		cv2.circle(i_colored_images,(i[0],i[1]),1,(0,0,255),2) # center point
		# cv2.putText(i_colored_images, str(count+1), (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
	cv2.imshow('coins', i_colored_images)
	# cv2.imshow('blur', blur)
	cv2.waitKey(10)
	newImgName = "dest/image"+'{:03d}'.format(globalcount)+".jpg"
	cv2.imwrite(newImgName, i_colored_images)
	globalcount += 1

def handleVideo():
	global frame, gray, video_capture




	# cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)
	# cv2.namedWindow('gray', cv2.WINDOW_KEEPRATIO)

	while True:
		if not video_capture.isOpened():
			print('Unable to load camera.')
			sleep(5)
			pass
		# Capture frame-by-frame
		ret, frame = video_capture.read()

		h, w, c = frame.shape
		x = 500
		y = w / (h/x)
		frame = cv2.resize(frame, (round(y), x))

		pprint('dentro')


		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		detectCircles(gray, frame, g_minDist, g_dp)


# image = cv2.imread("coins2.jpg", cv2.IMREAD_COLOR)
# cimg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.namedWindow("coins", cv2.WINDOW_KEEPRATIO)
#
cv2.createTrackbar('Parameter1','coins',5000,30000,parameter2)
cv2.createTrackbar('Parameter2','coins',0,100,parameter1)
cv2.createTrackbar('minradius','coins',1,100,radiusminfunc)
cv2.createTrackbar('maxradius','coins',1,500,radiusmaxfunc)
cv2.createTrackbar('cannymin','coins',1,100,cannyminfunc)
cv2.createTrackbar('cannymax','coins',1,500,cannymaxfunc)
# print('teste')
while True:
	# cv2.imshow("coins", image.astype('uint8'))
	key = cv2.waitKey(1) & 0xFF
	if key == ord("d"):

		video_capture = cv2.VideoCapture("IMG_1460.MOV")
		handleVideo()

		# detectCircles(cimg, image, g_minDist, g_dp)
	elif key == ord("e"):
		handleVideo()
		# detectCircles(cimg, image, g_minDist, g_dp)
	elif key == ord("c"):
		break
cv2.destroyAllWindows()
