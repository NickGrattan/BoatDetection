import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import os
import datetime
from shutil import copyfile, move
import tfdeploy as td

# suppress warning about not compiled for FMA instruction set
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

path = 'C:/BoatsV2/'
positives = path + 'Dataset/Positive/'
negatives = path + 'Dataset/Negative/'
cnnmodel = path + 'Model/isaboat'
root_logdir = path + 'Logs/'

std_image_size_x = 60
std_image_size_y = 60
channels = 3

Xlist = []
Ylist = []

def normalizeimage(img):
    img = cv2.resize(img, (std_image_size_x, std_image_size_y), cv2.INTER_AREA) 
    norm_image = np.zeros((std_image_size_x, std_image_size_y))
    norm_image = cv2.normalize(img, norm_image, 0, 255, cv2.NORM_MINMAX)
    # blur, but still maintain edges
    norm_image = norm_image.flatten()
    return norm_image

def addclassification(ispositive):
    if ispositive:
        Ylist.append(1)
    else:
        Ylist.append(0)

def rotateImage(img, angle):
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2,rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols,rows))

def processimage(fname, ispositive):
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    norm_image = normalizeimage(img)
    Xlist.append(norm_image)
    addclassification(ispositive)

def addextraimages(imgpaths, imgsents, x_training_list, y_training_list):
    # adds only to training set
    for idx in training_idx:
        if imgsents[idx] == 1:
            img = cv2.imread(imgpaths[idx], cv2.IMREAD_COLOR)
            # add flipped image
            norm_image = normalizeimage(cv2.flip(img, 1))
            x_training_list.append(norm_image)
            y_training_list.append(imgsents[idx])
            noise = cv2.randn(norm_image,(0),(99))
            x_training_list.append(norm_image + noise)
            y_training_list.append(imgsents[idx])
            # add -ve noise to image
            noise = cv2.randn(norm_image,(-99),(0))
            x_training_list.append(norm_image - noise)
            y_training_list.append(imgsents[idx])
            # add +ve rotated image
            rotpos = rotateImage(img, 15)
            rotnormimg = normalizeimage(rotpos)
            x_training_list.append(rotnormimg)
            y_training_list.append(imgsents[idx])
            # add +ve rotated image
            rotneg = rotateImage(img, -15)
            rotnormimg = normalizeimage(rotneg)
            x_training_list.append(rotnormimg)
            y_training_list.append(imgsents[idx])
    return (x_training_list, y_training_list)

print("Loading images...")
imgpaths = []
imgsents = []
positivecount = 0

# get list of positive images (boats)
for file in os.listdir(positives):
    if file.endswith(".jpg"):
        processimage(os.path.join(positives, file), True)
        positivecount += 1
        imgpaths.append(os.path.join(positives, file))
        imgsents.append(1)

# get list of negative images (not boats)
negativecount = 0
for file in os.listdir(negatives):
    if file.endswith(".jpg"):
        processimage(os.path.join(negatives, file), False)
        negativecount += 1
        imgpaths.append(os.path.join(negatives, file))
        imgsents.append(0)

Xd = np.array(Xlist, dtype=np.float32)
yd = np.array(Ylist, dtype=np.int)

# split into test and train
samples = Xd.shape[0]

indices = np.random.permutation(samples)
pc80 = int(samples * 80 / 100)
pc10 = int((samples - pc80) / 2)
training_idx, test_idx, validate_idx = indices[:pc80], indices[pc80 + 1:pc80 + pc10], indices[pc80 + pc10 + 1:]
X_training, X_test, X_valid = Xd[training_idx,:], Xd[test_idx,:], Xd[validate_idx,:]
y_training, y_test, y_valid = yd[training_idx], yd[test_idx], yd[validate_idx]

print("All History: Samples: {} Boats: {}".format(positivecount + negativecount, positivecount))
print("Train History: Samples: {} Boats: {}".format(len(y_training), y_training.sum()))
print("Test History: Samples: {} Boats: {}".format(len(y_test), y_test.sum()))
print("Validate History: Samples: {} Boats: {}".format(len(y_valid), y_valid.sum()))

D = std_image_size_x * std_image_size_y # number of features
K = 2                               # number of classes

### get indexes of positives/negatives from the training set
posind = np.where(y_training == 1)
negind = np.where(y_training == 0)

def get_next_training_batch(batchsize):
    batchhalf = int(batchsize / 2)
    possample = np.random.choice(posind[0], batchhalf)
    negsample = np.random.choice(negind[0], batchhalf)
    allsample = np.concatenate((possample, negsample))
    return (X_training[allsample], y_training[allsample])

#########################
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

height = std_image_size_x
width = std_image_size_y

n_inputs = height * width * channels

conv1_fmaps = 64
conv1_ksize = 4
conv1_stride = 2
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 4
conv2_stride = 2
conv2_pad = "SAME"
conv2_dropout_rate = 0.25

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 2

n_epochs = 30
batch_size = 50 
learning_rate = 0.001

now = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
logdir = "{}/run-{}/".format(root_logdir, now)

reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")

conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
fc0 = tf.layers.dense(conv1, n_fc1 * 5, activation=tf.nn.relu, name="fc0")
conv2 = tf.layers.conv2d(fc0, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

fc0a = tf.layers.dense(conv2, n_fc1 * 5, activation=tf.nn.relu, name="fc0a")
conv2a = tf.layers.conv2d(fc0a, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2a")

fc2 = tf.layers.dense(conv2a, n_fc1 * 5, activation=tf.nn.relu, name="fc2")
conv3 = tf.layers.conv2d(fc2, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv3")

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    factor = 4
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * factor])

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

best_loss_val = np.infty
check_interval = 100
checks_since_last_progress = 0
max_checks_without_progress = 150
best_model_params = None 

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

acc_summary = tf.summary.scalar('ACC', accuracy)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
       for iteration in range(X_training.shape[0] // batch_size + 1):
            X_batch, y_batch = get_next_training_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if iteration % check_interval == 0:
                loss_val = loss.eval(feed_dict={X: X_test, y: y_test})
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
       acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
       acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
       print("Epoch {}, train accuracy: {:.4f}%, test accuracy: {:.4f}%, test best loss: {:.6f}".\
            format(epoch, acc_train * 100, acc_test * 100, best_loss_val))

       summary_str = acc_summary.eval(feed_dict={X: X_test, y: y_test})
       file_writer.add_summary(summary_str, epoch)

       if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping!")
            break

    if best_model_params:
        restore_model_params(best_model_params)
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Final accuracy on test set:", acc_test)
    acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
    print("Final accuracy on validation set:", acc_valid)
    save_path = saver.save(sess, cnnmodel)
    print("Model saved")



