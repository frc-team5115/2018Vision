#import everything we need
import socket
import time
import cv2
import numpy as np
import os

#grab log data for debug
import logging
logging.basicConfig(level=logging.DEBUG)

#define our UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UDP_IP = "10.51.15.2"
UDP_PORT = 5800

#assign opencv camera to camera0
boilercam = cv2.VideoCapture(0)
#set propid(3, meaning camera width) to 160
boilercam.set(3, 160)
#set propid(4, meaing camera height) to 120
boilercam.set(4, 120)

counter = 0

#finding the Boiler 
def getOffsetsBoiler():
	#grab frame
	ret, frame = boilercam.read()
	#extract frame data
	height, width, channels = frame.shape
	#convert our color data from standard RGB to HSV
	#HSV stands for hue saturation value
	#see here https://en.wikipedia.org/wiki/HSL_and_HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
	#specify the lowest and highest colors we could possibly want
	lower = np.array([35, 100, 150])
	upper = np.array([75, 255, 220])
	#take out any frame data that IS NOT within our defined range
	thresh = cv2.inRange(hsv, lower, upper)
	#create contours, see here 
	#http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html#contours-getting-started
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	#initialize some variables for later
	maxarea = 0
	centerx = 0
	centery = 0
	contour = 0
	
	pixelOffsetX = 0
	pixelOffsetY = 0
	#within our list of contours
	for c in contours:
		m = cv2.moments(c)
		#keep going down the list and find the biggest possible one.
		#once the biggest one is found, assume that is our target
		if m['m00'] > maxarea:
			centerx = m['m10'] / m['m00']
			centery = m['m01'] / m['m00']
			maxarea = m['m00']
			contour = c
	#camera offset to account for any perspective differences	
	pixelOffsetX = centerx - (width / 2)
	pixelOffsetY = -(centery - (height / 2))

	#print str(pixelOffsetY)
	#print str(pixelOffsetX)

	return pixelOffsetX, pixelOffsetY

lastTime = time.clock()

#great, now that we have our data that we got from the frame we captured...
while True:
	#grab our x y data from the boiler function
	x, y = getOffsetsBoiler()

	#print str(counter)
	counter += 1

	#print str(x)
	#print str(y)
	#print str(gear)
	sock.sendto(("x " + x), (UDP_IP, UDP_PORT))
	sock.sendto(("y " + y), (UDP_IP, UDP_PORT))

	nowTime = time.clock()
	#print 1 / (nowTime - lastTime)
	lastTime = time.clock()

