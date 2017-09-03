import time
import datetime
from threading import Thread,Lock
import sys

import cv2
import numpy as np
import pandas as p

try:
    from picamera.array import PiRGBArray
    from picamera import PiCamera
except ImportError:
    pass

# parameters....
resolution = (1024, 1008)
# defines a region of interest for the whole frame.  This crops land areas
# that are not of interest.
roi_x1 = 90
roi_x2 = 700
roi_y1 = 500
roi_y2 = 840
queue_backpressure = 100 # maximum queue length 
interval = 1        # time between captures
warmdownperiod = 30       # wait time for building mask and stabilising image

# frames and masks used for testing without a pi camera
testframepath = "C:/Boats/Images/"
testframes = ["20170815-124442-7302",
    "20170804-141819-2381",
    "20170816-080251-1751",
    "20170815-200017-20631",
    "20170815-140458-9678",
    "20170815-140218-9605",
    "20170815-134141-8984", # two sailing boats
    "20170815-131437-8189", # motor cruiser
    "20170815-114957-5681",  # canoes
    "20170815-092746-1661",  #spirit of kinsale
    "20170821-153703-0",  # single large whole image
    "20170821-195214-92"  # large number of contours
    ]

class Collector(Thread):
    
    def __init__(self, queue_raw):
        self.queue_raw = queue_raw

        self.operating_mode = 'dark'
        # is pycamera available?
        self.picamera = 'picamera' in sys.modules
        # start the thread
        super(Collector, self).__init__()

    def run(self):
        print('Collector Started')
        while (True):
            try:
                self.getpictures()
            except Exception as e:
                print("Error capturing camera:", e)

    def getpictures(self):
        frame_count = 0
        interval = 1
        if self.picamera:
            warmdown = warmdownperiod
            with PiCamera(resolution=resolution) as camera:
                # camera warm up
                time.sleep(5)
                # camera configure
                camera.rotation = 90
                
                rawCapture = PiRGBArray(camera)
                self.apply_camera_light_settings(camera)
                # mask for background movement detection
                self.mask = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
                # grab an image from the camera
                frame_count = 0
                for frame in camera.capture_continuous(rawCapture, format="bgr"):
                    # check if outside of recording time:
                    now = datetime.datetime.now()
                    now_time = now.time()
                    if now_time >= datetime.time(5,0) and now_time <= datetime.time(20,00):
                        # ok, within day hours...
                        rawimage = frame.array

                        # check camera settings for brightness etc.
                        if frame_count % 300 == 0:
                            self.vary_camera_settings(rawimage, camera)

                        # extract area of interest
                        rawimage = rawimage[roi_y1:roi_y2, roi_x1:roi_x2]
                        # do motion detection
                        frame_motion = self.mask.apply(rawimage)
                        kernel=np.ones((2,2), np.uint8)
                        #apply morphology mask to decrease noise
                        frame_motion_output = cv2.morphologyEx(frame_motion, cv2.MORPH_OPEN, kernel=kernel)
                        imgray = cv2.cvtColor(rawimage, cv2.COLOR_BGR2GRAY)
                        maskedimage = cv2.bitwise_and(frame_motion_output, frame_motion, mask=frame_motion_output)
                        pkg = (datetime.datetime.now(), rawimage, maskedimage)
                        queuelen = self.queue_raw.qsize()
                        if queuelen < queue_backpressure:    # reduce backpressure
                            if warmdown > 0:
                                warmdown -= 1
                            else:
                                self.queue_raw.put(pkg)
                            if queuelen > queue_backpressure / 2:
                                interval = 5
                            else:
                                interval = 1
                        else:
                            warmdown = 20
                    else:
                        print("Sleeping ... it's dark")
                        warmdown = warmdownperiod
                        time.sleep(300)
                    frame_count += 1
                    rawCapture.truncate(0)
                    time.sleep(interval)
        else:
            while True:
                for tf in testframes:
                    rawimage,maskedimage = self.gettestframe(tf)
                    pkg = (datetime.datetime.now(), rawimage, maskedimage)
                    self.queue_raw.put(pkg)
                    time.sleep(1)
                break

    def gettestframe(self, basename):
        """
        Gets a raw image (as from camera) and the masked image for testing
        """
        fmaskedimage = testframepath + 'maskedimage-{}.jpg'.format(basename)
        frawimage = testframepath + 'raw-{}.jpg'.format(basename)
        maskedimage = cv2.imread(fmaskedimage,0)
        rawimage = cv2.imread(frawimage)
        return (rawimage, maskedimage)

    def apply_camera_light_settings(self, camera):
        camera.exposure_mode = 'auto'
        camera.contrast = 10
        camera.brightness = 60
        camera.exposure_compensation = 1
        self.operating_mode = 'light'
        print("Camera operating mode:", self.operating_mode)

    def apply_camera_dark_settings(self, camera):
        camera.exposure_mode = 'auto'
        camera.contrast = 20
        camera.brightness = 80
        camera.exposure_compensation = 3
        self.operating_mode = 'dark'
        print("Camera operating mode:", self.operating_mode)

    def vary_camera_settings(self, rawimage, camera):
        # Dynamically Vary Camera Settings
        intensity_mean = rawimage.ravel().mean() #8 bit camera
        print("Intensity mean: ", intensity_mean, self.operating_mode)
        #adjust camera properties dynamically if needed
        if intensity_mean < 115 and  self.operating_mode == 'light':
            self.apply_camera_dark_settings(camera)
            time.sleep(1)
            self.mask = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
            print('Dark Mode Activated - Camera')
        if intensity_mean > 120 and  self.operating_mode == 'dark':
            self.apply_camera_light_settings(camera)
            time.sleep(1)
            self.mask = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
            print('Light Mode Activated - Camera')
