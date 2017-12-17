from threading import Thread,Lock

import cv2
import numpy as np
import pandas as p
import math
import requests
import json
import io
import pickle
import zipfile
import sys
import time

from GraphObj import Point, Rectangle

# parameters
smallestcontour = 5     # used to remove small contours
mindist = 1             # minimum distance beween contours when merging
minradius = 0
minshiparea = 100       # minimum area (== number of pixes) for a ship
maxshiparea = 175 * 125   
maxcontours = 4000

class Detector(Thread):

    def __init__(self, queue_raw, queue_detectors):
        self.queue_raw = queue_raw
        self.queue_detectors = queue_detectors

        # start the thread
        super(Detector, self).__init__()

    def run(self):
        print('Detector Started')
        while True:
            timestamp, rawimage, maskedimage = self.queue_raw.get()
            contours = self.getcontours(maskedimage)
            if len(contours) == 0:
                continue
            contours = self.mergecontours(contours)
            rectangles = self.processcontours(contours)
            boats, rectangles = self.candidateboats(rectangles, rawimage)
            if len(boats) > 0:
                self.queue_detectors.put((timestamp, rawimage, maskedimage, boats, rectangles))

    def getcontours(self, maskedimage):
        """
        Returns contours from masked image, remove small contours 
        """
        (thresh, im_bw) = cv2.threshold(maskedimage, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        edges = cv2.Canny(im_bw, 100, 200)
        im2, contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        newcontours = []

        for c in contours:
            area = cv2.contourArea(c)
            if area > smallestcontour:
                newcontours.append(c)
        if len(contours) > maxcontours:
            print("Noise, too many contours {}".format(len(contours)))
            return []
        return newcontours
    
    def mergecontours(self, contours):
        """
        Merges close-by contours
        """
        def buildcontourlist(contours):
            contourlist = []
            for c in contours:
                center, radius = cv2.minEnclosingCircle(c)
                contourlist.append((c, center, radius))
            return contourlist

        def getfirstcandidate(contourlist):
            for i in range(len(contourlist)):
                ith = contourlist[i]
                for j in range(i+1, len(contourlist)):
                    jth = contourlist[j]
                    # distance between two centers
                    d = math.hypot(abs(ith[1][0] - jth[1][0]), abs(ith[1][1]- jth[1][1]))
                    # subtract sum of two radii
                    d -= (ith[2] + jth[2])
                    if d < mindist:
                        return (int(round(d)), i, j)
            return (mindist + 1, 0, 0)

        def mergecontours(ith, jth, contours, contourlist):
            cont = np.vstack(contours[i] for i in [ith, jth])
            merged = cv2.convexHull(cont)
            contours = [c for i, c in enumerate(contours) if i not in [ith, jth]]
            contourlist = [c for i, c in enumerate(contourlist) if i not in [ith, jth]]
            contours.append(merged)
            center, radius = cv2.minEnclosingCircle(merged)
            contourlist.append((merged, center, radius))
            return (contours, contourlist)

        contourlist = buildcontourlist(contours)
        while True:
            d, ith, jth = getfirstcandidate(contourlist)
            if d > mindist:
                break
            contours, contourlist = mergecontours(ith, jth, contours, contourlist)
            time.sleep(.0001)
        return contours

    def processcontours(self, contours):
        """
        Further contour processing, returns list of rectangles surrounding boat candidates
        """
        # see if a contour is  contained in another
        toremove=[]
        for i, c1 in enumerate(contours):
            for j, c2 in enumerate(contours):
                if i != j:
                    m2 = cv2.moments(c2)
                    cX = int(m2["m10"] / m2["m00"])
                    cY = int(m2["m01"] / m2["m00"])
                    if cv2.pointPolygonTest(c1, (cX, cY), False) > 0:
                        toremove.append(j)
        toremove=set(toremove)
        contours = [c for i, c in enumerate(contours) if i not in toremove ]
        time.sleep(.0001)

        # remove contours whose bounding rectangle is in another contour's bounding rectangle
        rectangles=[]
        for i, c in enumerate(contours):
            x, y, w,h = cv2.boundingRect(c)
            rectangles.append(Rectangle(Point(x, y), Point(x + w, y + h)))
        time.sleep(.0001)
    
        toremove=[]
        for i, r1 in enumerate(rectangles):
            for j, r2 in enumerate(rectangles):
                if i != j:
                    if r1.contains(r2):
                        toremove.append(j)
        toremove=set(toremove)
        contours = [c for i, c in enumerate(contours) if i not in toremove]
        rectangles = [r for i, r in enumerate(rectangles) if i not in toremove]
        time.sleep(.0001)
        
        # merge adjacent bounding rectangles
        def mergerectangles(rects):
            for i, r1 in enumerate(rects):
                for j, r2 in enumerate(rects):
                    if i != j and r1.adjacent(r2):
                        rects = [r for rn, r in enumerate(rects) if rn != i and rn != j ]
                        rects.append(r1.merge(r2))
                        return (True, rects)
            return (False, rects)

        loop = True
        while loop:
            loop, rectangles = mergerectangles(rectangles)
            time.sleep(.0001)
        return rectangles                

    def candidateboats(self, rectangles, rawimage):
        """
        get list of candidate boat images from rectangles
        """
        boats=[]
        boatrectangles=[]
        for idx, r in enumerate(rectangles):
            roi = rawimage[r.top_left.y:r.bottom_right.y, r.top_left.x:r.bottom_right.x]
            # check that area of image is above minimum area of interest
            area = roi.shape[0] * roi.shape[1]
            if area > minshiparea and area < maxshiparea:
                boats.append(roi)
                boatrectangles.append(r)
        time.sleep(.0001)
        return (boats, boatrectangles)
