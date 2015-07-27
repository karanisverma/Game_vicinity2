import numpy as np
import cv2

cap = cv2.VideoCapture('yout.mp4')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
#r,h,c,w = 250,90,400,125  # simply hardcoded the values
r,h,c,w = 350,220,100,325  # simply hardcoded the values
track_window = (c,r,w,h)
count=0
# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
bx1=0
bx2=10
by1=0
by2=10
road_counter=0
while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        #print(track_window)
        x,y,w,h = track_window
        print(x)

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
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
            cv2.rectangle(frame, (bx1, by1), (bx1+bx2, by1+by2), (0, 255, 0), thickness=-1, lineType=8, shift=0)
            road_counter+=1
            
            #cv.Rectangle(img, pt1, pt2, color, thickness=-1, lineType=8, shift=0) 

        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',frame)
        #cv2.imshow('img4',gray)
        cv2.imshow('img3',edges)
        cv2.imshow('img5',gray)


        # define range of blue color in HSV
        lower_pink = np.array([115,0,150])
        upper_pink = np.array([179,255,255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        res = cv2.bitwise_and(frame,frame, mask= mask)
        cv2.imshow('res',res)
        #cv2.imshow('mask',mask)


        #cv2.imshow('hsv',hsv)
        count=count+1
        print(count)
        k = cv2.waitKey(60) & 0xff
        if k == 27 :
            break
       
    else:
        print("You are pretty much screwed up !")
        break
print("The Road Counter for the execution is : "+str(road_counter/2))
cv2.destroyAllWindows()
cap.release()



# import numpy as np
# import cv2

# cap = cv2.VideoCapture('yout.mp4')

# # take first frame of the video
# ret,frame = cap.read()

# # setup initial location of window
# r,h,c,w = 350,220,100,325  # simply hardcoded the values
# track_window = (c,r,w,h)

# # set up the ROI for tracking
# roi = frame[r:r+h, c:c+w]
# hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
# term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

# while(1):
#     ret ,frame = cap.read()

#     if ret == True:
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

#         # apply meanshift to get the new location
#         ret, track_window = cv2.meanShift(dst, track_window, term_crit)

#         # Draw it on image
#         x,y,w,h = track_window
#         #height, width, depth = img2.shape
#         img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        
#         # if(height>0 and width>0):
#         cv2.imshow('img2',frame)

#         k = cv2.waitKey(60) & 0xff
#         if k == 27:
#             break
#         else:
#             cv2.imwrite(chr(k)+".jpg",img2)

#     else:
#         break

# cv2.destroyAllWindows()
# cap.release()