import numpy as np
import pandas as pd
import cv2
import pickle

logfolder = "c:/ForClassification/Log2/"

class Frame():
    """
    Manages a single frame and associated data
    """

    def __init__(self,fname):
        self.load(fname)

    def load(self, thisdatetime):
        """
        Loads a frame from the given filename
        """
        tss = thisdatetime.strftime("%Y%m%d-%H%M%S")
        fname = logfolder + tss + ".pik"
        self.timestamp, self.rawimage, self.maskedimage, self.boats, self.rectangles = \
            pickle.load(open(fname, "rb"))

    def drawrectangles(self):
        """
        Draws rectangles on raw image and returns the image
        """
        ri = self.rawimage.copy()
        for r in self.rectangles:
            cv2.rectangle(ri, (r.top_left.x, r.top_left.y), (r.bottom_right.x, r.bottom_right.y), \
                (128, 0, 128), 1)
        return ri
