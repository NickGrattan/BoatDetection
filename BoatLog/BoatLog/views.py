"""
Routes and views for the flask application.
"""

from datetime import datetime, timedelta
from flask import render_template, send_file, request, session, jsonify
import io
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import cv2

from BoatLog import app
from BoatLog.frames import *

framedict = {}
std_image_size_x = 60
std_image_size_y = 60
historylen = 10

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template('index.html',
        title='Home Page',
        year=datetime.now().year,)

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template('contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.')

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template('about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.')

@app.route('/timeline', methods=['GET', 'POST'])
def timeline():
    """Renders the about page."""
    readframes()
    currentframe = len(framedict['allframes']) - 1
    if 'framenum' in request.args:
        currentframe = int(request.args.get('framenum'))
    currentframe = min(currentframe, len(framedict['allframes']) - 1)
    frame, currentframe = getframe(currentframe)
    return render_template('timeline.html',
        year=datetime.now().year,
        alldates= framedict['alldates'],
        numboats=len(frame.boats),
        currentframe=currentframe)

@app.route('/get_viewimage')
def get_viewimage():
    imgtype = int(request.args.get('type'))
    currentframe = int(request.args.get('currentframe'))
    frame, currentframe = getframe(currentframe)
    if imgtype == -1: # normal image
        img = frame.drawrectangles()
        # draw history positions
        width = 10
        for h in range(historylen + 1):
            historyframe, _ = getframe(currentframe - h)
            # only draw history for recent events
            time_difference_in_minutes = (frame.timestamp - historyframe.timestamp) / timedelta(minutes=1)
            if time_difference_in_minutes < 10:
                for r in historyframe.rectangles:
                    # find mid point in rectangle
                    cx = r.top_left.x + ((r.bottom_right.x - r.top_left.x) / 2)
                    #cy = r.top_left.y + ((r.bottom_right.y - r.top_left.y) / 2)
                    cy = r.bottom_right.y - width / 2
                    cv2.circle(img,(int(cx),int(cy)), width, (0,0,255), 1, cv2.LINE_AA)
            width -= 1
        ret, image_binary = cv2.imencode('.jpg',img)
    elif imgtype == -2: # masked image
        ret, image_binary = cv2.imencode('.jpg',frame.maskedimage)
    else:
        ret, image_binary = cv2.imencode('.jpg', normalizeimage(frame.boats[imgtype]))
    return send_file(io.BytesIO(image_binary),  mimetype='image/png')

@app.route('/framestatus')
def get_framestatus():
    """
    Returns the number of boats for the given frame
    """
    framenumber = request.args.get('framenum', 0, type=int)
    frame, framenumber = getframe(framenumber)
    df = gethistory(framenumber, historylen)
    history = df.to_html(escape=False)
    data = {'numboats':len(frame.boats), 'timestamp':'{:%X}'.format(frame.timestamp), \
            'framenumber' : framenumber, 'history' : history}
    return jsonify(data)

def readframes():
    if 'allframes' not in framedict:
        framedict['allframes'] = {}
    readtimeline()
    framedict['alldates'] = datelist(framedict['allframes'])
    framedict['allframedates'] = sorted(framedict['allframes'].keys())

def readtimeline():
    """
    Reads the available time line (list of pik files)
    """
    pikfiles = [f for f in listdir(logfolder) if isfile(join(logfolder, f))]
    for file in pikfiles:
        dt = datetime.strptime(file[0:-4], "%Y%m%d-%H%M%S")
        if dt not in framedict['allframes']:
            framedict['allframes'][dt] = None

def datelist(allframes):
    """
    Gets list of dates from allframes dictionary
    """
    dateset = set([dt.date() for dt in allframes.keys()])
    return sorted(list(dateset))

def getframe(framenumber):
    """
    Returns frame from cache, reads from disk if required
    """
    if framenumber >= len(framedict['allframedates']):
        readframes()
        if framenumber >= len(framedict['allframedates']):
            framenumber = len(framedict['allframedates']) - 1
    dt = framedict['allframedates'][framenumber]
    frame = framedict['allframes'][dt]
    if frame is None:
        frame = Frame(dt)
        framedict['allframes'][dt] = frame
    return (frame, framenumber)

def gethistory(currentframe, historylen):
    """
    Gets the recent history from the current frame, back 'historylen' frames
    """
    if 'framehistory' not in framedict:
        framedict['framehistory'] = {}
    if currentframe in framedict['framehistory']:
        return framedict['framehistory'][currentframe]
    pd.set_option('display.max_colwidth', -1)
    lstframenums = []
    lsttimestamp = []
    lstnumboats = []
    lstboaturls = []
    lstimgs = []
    for i in range(currentframe, currentframe - historylen, -1):
        cf, fn = getframe(i)
        lstframenums.append(fn)
        lsttimestamp.append('{:%X}'.format(cf.timestamp))
        lstnumboats.append(len(cf.boats))
        imgs = ""
        for j in range(len(cf.boats)):
            thisimage = "<img src='./get_viewimage?type=" + str(j) + "&currentframe=" + str(i) + "'/>&nbsp;&nbsp;"
            imgs += thisimage
        lstimgs.append(imgs)
    data = {'FrameNum':lstframenums, 'Time':lsttimestamp, 'NumBoats':lstnumboats, 'Boats' : lstimgs}
    df = pd.DataFrame(data)
    framedict['framehistory'][currentframe] = df
    return df

def normalizeimage(img):
    img = cv2.resize(img, (std_image_size_x, std_image_size_y), cv2.INTER_AREA) 
    norm_image = np.zeros((std_image_size_x, std_image_size_y))
    norm_image = cv2.normalize(img, norm_image, 0, 255, cv2.NORM_MINMAX)
    return norm_image
