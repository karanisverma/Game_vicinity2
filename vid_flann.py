# Initiate SIFT detector
import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('C:\Users\Karma_is_Bitch\Desktop\sample.jpg',0)          # queryImage
#img1 = cv2.imread('C:\Users\Karma_is_Bitch\Downloads\work\College\Final Year Project\Resource\Test Sample\i8.jpg',0)          # queryImage
vc = cv2.VideoCapture(0)
# vc = cv2.VideoCapture('estvid.flv')
f = open('data.txt', 'wb')

#img2 = cv2.imread('C:\Users\Karma_is_Bitch\Downloads\work\College\Final Year Project\Resource\Test Sample\wallhaven-46287.jpg',0) # trainImage
while(vc.isOpened()):
	# Initiate SIFT detector
	orb = cv2.ORB()
	ret, img2 = vc.read()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)
	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	img3 = cv2.drawKeypoints(img1,kp1,color=(0,255,0), flags=0)
	img4 = cv2.drawKeypoints(img2,kp2,color=(0,255,0), flags=0)
	# Draw first 10 matches.
	# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)
	
	#f.write(str(matches))
	# plt.imshow(img3),plt.show()
	# plt.imshow(img4),plt.show()
	cv2.imshow('clip',img4)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

	# plt.subplot(121),plt.imshow(img3,cmap = 'gray')
	# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	# plt.subplot(122),plt.imshow(img4,cmap = 'gray')
	# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

	# plt.show()
f.close()