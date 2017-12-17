from flask import Flask, url_for, json, jsonify, request, Response
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import os
import io
import pickle
import zipfile
import time
import shutil

# suppress warning about not compiled for FMA instruction set
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

std_image_size_x = 60
std_image_size_y = 60

testfolder_ptv = "C:/ForClassification/Test/Positive2/"
testfolder_neg = "C:/ForClassification/Test/Negative2/"
logfolder = "C:/ForClassification/Log2/"

def normalizeimage(img):
    img = cv2.resize(img, (std_image_size_x, std_image_size_y), cv2.INTER_AREA) 
    norm_image = np.zeros((std_image_size_x, std_image_size_y))
    norm_image = cv2.normalize(img, norm_image, 0, 255, cv2.NORM_MINMAX)
    norm_image = norm_image.flatten()
    return norm_image

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

@app.route("/TestBoats", methods=['POST'])
def api_testboats():
    """
    Receives boat images as list of numpy arrays
    """
    zbs = io.BytesIO(request.data)
    # unzip pickle
    with zipfile.ZipFile(zbs, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        pb = zip_file.read("Pik")
    pbs = io.BytesIO(pb)
    boatlist = pickle.load(pbs)
    
    print("Number of boats to test: Request Length:", len(boatlist), len(request.data))
    Xclasslist = []
    Yclasslist = []
    for boat in boatlist:
        boatimg = np.array(boat)
        boatimg = boatimg.astype(np.uint8)
        norm_image = normalizeimage(boatimg)
        Xclasslist.append(norm_image)
        Xclassd = np.array(Xclasslist, dtype=np.float32)

    # predict those to be classified
    Z = logits.eval(feed_dict={'inputs/X:0': Xclassd}, session=sess)
    y_pred = np.argmax(Z, axis=1)
    print("Predications:", y_pred)

    data = json.dumps(list(y_pred.tolist()))
    # dump images to test folders for accuracy testing
    timestr = time.strftime("%Y%m%d-%H%M%S")
    for i, p in enumerate(y_pred.tolist()):
        if p == 1:
            fname = testfolder_ptv + "{}-{}.jpg".format(timestr, i)
        else:
            fname = testfolder_neg + "{}-{}.jpg".format(timestr, i)
        cv2.imwrite(fname, boatlist[i])
    # end dump
    resp = Response(data, status=200, mimetype='application/json')
    return resp

@app.route("/FramePackage", methods=['POST'])
def api_framepackage():
    """
    Receives a package with frame images, boats and rectangles
    """
    print("Package: Request Length:", len(request.data))
    zbs = io.BytesIO(request.data)
    # unzip pickle
    with zipfile.ZipFile(zbs, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        pb = zip_file.read("Pik")
    pbs = io.BytesIO(pb)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    pbs.seek(0)
    with open(logfolder + timestr + ".pik", 'wb') as f:
        shutil.copyfileobj(pbs, f)
    data = json.dumps("OK")
    resp = Response(data, status=200, mimetype='application/json')
    return resp

if __name__ == "__main__":
    dnnmodel = 'C:/BoatsV2/Model/isaboat'
    dnnmetamodel = 'C:/BoatsV2/Model/isaboat.meta'

    # setup tensorflow session    
    reset_graph()
    init = tf.global_variables_initializer()
    saver = tf.train.import_meta_graph(dnnmetamodel)
    sess = tf.Session()
    saver.restore(sess,dnnmodel)
    graph = tf.get_default_graph()
    logits = graph.get_tensor_by_name('output/output/BiasAdd:0')

    app.run(host='0.0.0.0', debug=True)
