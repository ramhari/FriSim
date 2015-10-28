import numpy as np
import cv2

class App:
    def __init__(self, fn):
        self.img = cv2.imread(fn)
        self.DisplayandSave('Input', self.img)
        h, w = self.img.shape[:2]
        self.markers = np.zeros((h, w), np.int32)
        self.colors = np.int32( list(np.ndindex(2, 2, 2)) ) * 255

        #Thresholding
        self.gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.ret, self.thresh = cv2.threshold(self.gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.DisplayandSave('Threshold', self.thresh)

        # noise removal
        self.kernel = np.ones((3,3),np.uint8)
        self.opening = cv2.morphologyEx(self.thresh,cv2.MORPH_OPEN,self.kernel, iterations = 2)
        self.closing = cv2.morphologyEx(self.opening,cv2.MORPH_OPEN,self.kernel, iterations = 2)
        self.DisplayandSave('After noise removal', self.closing)

        # Marker labelling
        self.ret, self.markers = cv2.connectedComponents(self.closing)
        self.markers = self.markers+1

    def DisplayandSave(self,name,img):
        cv2.imshow(name,img)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
        elif k == ord('s'): # wait for 's' key to save and exit
            cv2.imwrite(name,img)
            cv2.destroyAllWindows()

    def findContours(self):
        _ , self.contours, _ = cv2.findContours(self.closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.img, self.contours, -1, (0,255,0), 3)
        self.DisplayandSave('Image with contours',self.img)
        for i in range(len(self.contours)):
            cnt = self.contours[i]
            #print cnt
            area = cv2.contourArea(cnt)
            print "Object #",i
            print "Area: ",area
            perimeter = cv2.arcLength(cnt,True)
            print "Perimeter: ",perimeter
            x,y,w,h = cv2.boundingRect(cnt)
            print "Rectangle Dims:",x,y,w,h
            cv2.rectangle(self.img,(x,y),(x+w,y+h),(0,255,0),2)
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            print "Circle Dims:",x,y,radius
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(self.img,center,radius,(0,255,0),2)  
            print cnt
            mean, eigenvectors = cv2.PCACompute(cnt, np.mean(cnt, axis=0).reshape(1,-1))
            print mean,eigenvectors
            self.DisplayandSave('I mage with bounding box',self.img)

    def watershed(self):
        m = self.markers.copy()
        cv2.watershed(self.img,m)
        self.img[m == -1] = [255,0,0]
        overlay = self.colors[np.maximum(m, 0)]
        vis = cv2.addWeighted(self.img, 0.5, overlay, 0.5, 0.0, dtype=cv2.CV_8UC3)
        self.DisplayandSave('watershed', vis)

    def run(self):
        self.findContours()

fn = 'pressuremapbefore.jpg'
print "version",cv2.__version__
App(fn).run()
