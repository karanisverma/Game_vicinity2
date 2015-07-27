import cv2
import numpy as np

video_capture = cv2.VideoCapture('yout.mp4')
print(video_capture.isOpened())
bx1=0
bx2=10
by1=0
by2=10
while(video_capture.isOpened()):
	ret, img = video_capture.read()							#img = cv2.imread('dave.jpg')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 10, 27, 27)
	edges = cv2.Canny(gray,50,150,apertureSize = 3)
	minLineLength = 100
	maxLineGap = 20
	lines = cv2.HoughLinesP(edges,1,np.pi/90,100,minLineLength,maxLineGap)
	for x1,y1,x2,y2 in lines[0]:
		#print(lines)
		slope=(y2-y1)/(x2-x1)
		
		if(slope==0) :
			continue
		#cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
		if(slope == -1):
			bx1=x1
			by1=y1
		if(slope==1):
			bx2=x2
			bx2=y2
		cv2.rectangle(img, (bx1, by1), (bx1+bx2, by1+by2), (0, 255, 0), thickness=-1, lineType=8, shift=0)
		
		#cv.Rectangle(img, pt1, pt2, color, thickness=-1, lineType=8, shift=0) 

	cv2.imshow('Video', img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
							#cv2.imwrite('houghlines5.jpg',img)