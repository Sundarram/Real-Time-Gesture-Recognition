# -*- coding: cp1252 -*-
import pickle
import numpy as np
from numpy import sqrt, arccos, rad2deg
import cv2
# local modules
from video import create_capture
from common import clock, draw_str

class Tracking:

    def __init__(self):
        self.debugMode = True

        self.camera = cv2.VideoCapture(0) 
        
        #Setting resolution of camera
        self.camera.set(3,640)
        self.camera.set(4,480)
        
        self.posPre = center_x  #Setting Initial position of the centroid
        self.prelen = 1
        #Dictionary to store our data
        self.Data = {
                     "cursor" : (0, 0),
                     "hulls" : 0, 
                     "Contours": 0,
                     "cursor2" : (0, 0),
                     }
        #Updating every time
        self.lastData = self.Data

        #Loading configuration for Skin filter from file config
        try:  self.Vars = pickle.load(open(".config", "r"))
        except:
            print "Config file («.config») not found."
            exit()


        #Print text
        self.addText = lambda image, text, point:cv2.putText(image,text, point, cv2.FONT_HERSHEY_PLAIN, 1.0,(255,255,255))     
#----------------------------------------------------------------------  
        #Always 
        while (self.camera.isOpened()):
            self.run()                  #Process the image
            self.recognize()            #Recognize the gestures
            
            if self.debugMode:
                if cv2.waitKey(1) == 27: break
            
        self.camera.release()           #Release the camera
        cv2.destroyAllWindows()

#----------------------------------------------------------------------
    def run(self):
        ret, im = self.camera.read()
        im = cv2.flip(im, 1)
        self.imOrig = im.copy()
        self.imNoFilters = im.copy()
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)          #Used for Masking
        #Applies smooth
        im = cv2.blur(im, (self.Vars["smooth"], self.Vars["smooth"]))
        
        #Applies skin color filter
        filter_ = self.skinfilter(im)

        #Applies erode
        filter_ = cv2.erode(filter_,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.Vars["erode"], self.Vars["erode"])))           
        
        #Applies dilate
        filter_ = cv2.dilate(filter_,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.Vars["dilate"], self.Vars["dilate"])))
        
        
        #Displays the binary image
        if self.debugMode:
            cv2.imshow("Skin Filter Image", filter_)
            cv2.moveWindow("Skin Filter Image",0,0)
        
        #Obtain the contours
        contours, hierarchy = cv2.findContours(filter_,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        
        self.Data["Contours"] = len(contours);              # No. of contours
        
        #Eliminating small areas (unnecessary)
        allIndex = []
        for index in range(len(contours)):
            area = cv2.contourArea(contours[index])
            if area < 5e3: allIndex.append(index)
        allIndex.sort(reverse=True)
        for index in allIndex: contours.pop(index)

        #If no contours, ends here
        if len(contours) == 0: return
      
        allIndex = []
        index_ = 0

        
    #----------------------------------------------------------------------  
        
        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        
        mask = np.zeros(filter_.shape,np.uint8)
        cv2.drawContours(mask,[cnt],0,255,-1)
        mean = cv2.mean(im, mask = mask)
        
        M2 = cv2.moments(cnt)
        centrix = int(M2['m10']/M2['m00'])
        centriy = int(M2['m01']/M2['m00'])
        cv2.circle(mask, (centrix, centriy), 20, (0,255,255), 10)       #Black circles in Mask Image 
        self.Data["cursor2"] = (centrix, centriy)
            
        #if self.debugMode: cv2.imshow("MASK", mask)
        #We visit each contour      
        for cnt in contours:
            tempIm = im.copy()
            tempIm = cv2.subtract(tempIm, im)
            
            #To find convexities (fingers)
            hull = cv2.convexHull(cnt)
            self.last = None
            self.Data["hulls"] = 0
            for hu in hull:
                if self.last == None: cv2.circle(tempIm, tuple(hu[0]), 10, (0,0,255), 5)
                else:
                    distance = self.distance(self.last, tuple(hu[0]))
                    if distance > 40:  
                        self.Data["hulls"] += 1
                        # Red circles representing endpoints of convex hulls
                        cv2.circle(tempIm, tuple(hu[0]), 10, (0,0,255), 5)
                self.last = tuple(hu[0])

            
            #Finding Centroid using moments
            M = cv2.moments(cnt)
            centroid_x = int(M['m10']/M['m00'])
            centroid_y = int(M['m01']/M['m00'])
            cv2.circle(tempIm, (centroid_x, centroid_y), 10, (0,255,255), 10) #Centroid of Contour
##            print 'center_x:' + repr(center_x)
##            print 'selfpospre:' + repr(self.posPre)                           # FOR DEBUG PURPOSE
##            #print 'center_y:' + repr(center_y)
##            #cv2.circle(tempIm, (center_x, center_y), 10, (255,255,255), 10)
            self.Data["cursor"] = (centroid_x, centroid_y)
            
            #To find the convexity defects (spaces between fingers)  - useful to calculate no. of fingers on screen
            hull = cv2.convexHull(cnt,returnPoints = False)

            defects = cv2.convexityDefects(cnt,hull)
            if defects == None: return
          
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                if d > 1000 :
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])

                    #Lines between convexities and defects
                    cv2.line(tempIm, start, far, [255, 0, 0], 5) 
                    cv2.line(tempIm, far, end, [255, 0, 0], 5)
            self.imOrig = cv2.add(self.imOrig, tempIm)
            
            index_ += 1


        #Display the state of the actual data
        self.debug()
        if self.debugMode:
            cv2.imshow("Camera Mode", self.imOrig)
            cv2.moveWindow("Camera Mode",650,0)

        
    #----------------------------------------------------------------------
    def distance(self, cent1, cent2):
        #Returns the distance between two points
        x = abs(cent1[0] - cent2[0])
        y = abs(cent1[1] - cent2[1])
        d = sqrt(x**2+y**2)
        return d

    #----------------------------------------------------------------------
    def skinfilter(self, im):
        #Applies skin color filter
        UPPER = np.array([self.Vars["upper"], self.Vars["filterUpS"], self.Vars["filterUpV"]], np.uint8)
        LOWER = np.array([self.Vars["lower"], self.Vars["filterDownS"], self.Vars["filterDownV"]], np.uint8)
        hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        filter_im = cv2.inRange(hsv_im, LOWER, UPPER)
        return filter_im

    #----------------------------------------------------------------------
    def debug(self):
        #Debug mode
        yPos = 10
        if self.debugMode: self.addText(self.imOrig, "Debug", (yPos, 20))
        pos = 50
        #Display all the details
        for key in self.Data.keys():
            if self.debugMode: self.addText(self.imOrig, (key+": "+str(self.Data[key])), (yPos, pos))
            pos += 20
        
    #----------------------------------------------------------------------
    def recognize(self):
        #Importing the left and right indicator images
        leftindicator=cv2.imread('C:/Python27/left.jpg')
        rightindicator=cv2.imread('C:/Python27/right.jpg')
        pos=self.Data["cursor2"][0]
        #counter = self.Data["Contours"]
        posPre = self.posPre
        npos = np.subtract(pos,posPre)
        self.posPre = pos
        # if centroid changes position more than 20 pop up the respective indicator 
        if npos < -20:
            
            cv2.imshow('Indicator',leftindicator)
            cv2.moveWindow("Indicator",350,200)
            cv2.waitKey(1000)
            cv2.destroyWindow("Indicator")
            
        elif npos > 20:
            
            cv2.imshow('Indicator',rightindicator)
            cv2.moveWindow("Indicator",350,200)
            cv2.waitKey(1000)
            cv2.destroyWindow("Indicator")
#----------------------------------------------------------------------
def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    print repr(len(rects))
    rects[:,2:] += rects[:,:2]
    return rects
#----------------------------------------------------------------------

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__=='__main__':
    import sys, getopt
    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    cascade = cv2.CascadeClassifier(cascade_fn)
    cam = create_capture(video_src, fallback='synth:bg=../cpp/lena.jpg:noise=0.05')
    
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.equalizeHist(gray1)
    
    t = clock()
    rects = detect(gray2, cascade)
    #print (repr(len(rects)))
    vis = img.copy()
    draw_rects(vis, rects, (0, 255, 0))
    # If a Face is detected, Start Gesture Recognition
    if len(rects) == 1:
        draw_str(vis, (20, 20), 'Face Detected!')
        cv2.imshow('facedetect', vis)
        cv2.waitKey(1000)
        cv2.destroyWindow("facedetect")
        c1 = (rects[0][2] - rects[0][0])/2
        center_x = c1 + rects[0][0]
        c2 = (rects[0][3] - rects[0][1])/2
        center_y = c1 + rects[0][1]
        Tracking()
        
