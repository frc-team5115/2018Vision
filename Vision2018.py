#import everything we need
import socket
import time
import cv2 as cv
import numpy as np
import os

#grab log data for debug
import logging
logging.basicConfig(level=logging.DEBUG)

#define our UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UDP_IP = "10.51.15.2"
UDP_PORT = 5803
roborio = (UDP_IP, UDP_PORT)
#define video stream
stream = 10.51.15.30:8080/?action=stream
#assign opencv camera to URL
boilercam = cv.VideoCapture(stream)
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
	hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
	#specify the lowest and highest colors we could possibly want
	lower = np.array([35, 100, 150])
	upper = np.array([75, 255, 220])
	#take out any frame data that IS NOT within our defined range
	thresh = cv.inRange(hsv, lower, upper)
	#create contours, see here 
	#http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html#contours-getting-started
	contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	#initialize some variables for later
	maxarea = 0
	centerx = 0
	centery = 0
	contour = 0
	
	pixelOffsetX = 0
	pixelOffsetY = 0
	#within our list of contours
	for c in contours:
		m = cv.moments(c)
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
	s
	x = "x " + str(x)
	y = "y " + str(y)
	
	sock.sendto(x, roborio)
	sock.sendto(y, roborio)

	nowTime = time.clock()
	#print 1 / (nowTime - lastTime)
	lastTime = time.clock()

