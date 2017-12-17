from threading import Thread,Lock

import cv2
import numpy as np
import pandas as p
import platform
import io
import pickle
import zipfile
import requests

if platform.system() == 'Windows':
    dumppath = 'c:/temp/Boats/Dump/'
else:
    dumppath = '/home/pi/Boats/Dump/'

restUrlPackage = "http://192.168.0.26:5000/FramePackage"    # URL for REST service sending frame data
restUrlTest = "http://192.168.0.26:5000/TestBoats"    # URL for REST service for boat detection

class Processor(Thread):

    def __init__(self, queue_detectors):
        self.queue_detectors = queue_detectors

        # start the thread
        super(Processor, self).__init__()

    def run(self):
        print('Processor Started')
        while True:
            timestamp, rawimage, maskedimage, boats, rectangles = self.queue_detectors.get()

            # make REST call use CNN to classify images as boat/not boats
            # pickle boats to binary stream
            pbs = io.BytesIO()
            pickle.dump(boats, pbs)

            # zip the pickle buffer https://stackoverflow.com/questions/2463770/python-in-memory-zip-library
            zbs = io.BytesIO()
            with zipfile.ZipFile(zbs, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                zip_file.writestr("Pik", pbs.getvalue())
            response=requests.post(restUrlTest, data=zbs.getvalue(), headers={'Content-Type': 'application/octet-stream'}) #, timeout=10)
            classifications = response.json()

            # make package of images identified as boats
            actualboats = [b for i, b in enumerate(boats) if classifications[i] == 1]
            actualrectangles = [r for i, r in enumerate(rectangles) if classifications[i] == 1]
            if len(actualboats) > 0:
                tss = timestamp.strftime("%Y%m%d-%H%M%S")
                print(tss, "Detected Boats:", len(actualboats))

                # send package of images to server...
                pbs = io.BytesIO()
                pickle.dump((timestamp, rawimage, None, actualboats, actualrectangles), pbs)
                zbs = io.BytesIO()
                with zipfile.ZipFile(zbs, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    zip_file.writestr("Pik", pbs.getvalue())

                response=requests.post(restUrlPackage, data=zbs.getvalue(), headers={'Content-Type': 'application/octet-stream'}) #, timeout=10)
                status = response.json()
                if status != 'OK':
                    print("Processor call to server failed:", status)
