#!/usr/bin/env python3
from tensorflow import keras
import cv2
import numpy as np
from common import *

def load_model(model_path):

    with open(model_path + "/model.json") as json_file:
        json_config = json_file.read()

    model = keras.models.model_from_json(json_config)
    model.load_weights(model_path + "/model_weights.h5")

    return model



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    if width is not None and height is not None:
        dim = (width,height)

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized



def convert_data_to_image(x_data, y_data):
    # Input.
    image = np.reshape(x_data, [HEIGHT, WIDTH, CHANNEL])


    # Labels
    labels = []

    n_row, n_col, _ = y_data.shape

    for row in range(n_row):
        for col in range(n_col):
            
            d = y_data[row, col]
            # If cash note in the grid cell
            if d[0] < DETECTION_PARAMETER:
                continue
            # print(d[1:5])
            # Convert data.
            bx, by, bw, bh = d[1:5]

            w = bw * WIDTH
            h = bh * HEIGHT
            x = col * GRID_WIDTH + (bx * GRID_WIDTH - w/2)
            y = row * GRID_HEIGHT + (by * GRID_HEIGHT - h/2)

            s = CLASSES[np.argmax(d[5:])]

            # print([d[0],x,y,w,h,s])
            # labels
            labels.append([d[0],x,y,w,h,s])

    return image, labels



def render_with_labels(image, labels, display = False):
    colors = {
        "RM50" : (0,255,0),
        "RM1" : (255,0,0),
        "RM10" : (0,0,255),
        "RM20" : (0,128,255),
        "RM100" : (255,255,255),
    }

    for label in labels:
        # cv2.rectangle(image,(0,0),(5,5),(0,255,0),2)
        cv2.rectangle(image, (int(label[1]),int(label[2])), (int(label[1]+label[3]),int(label[2]+label[4])), colors[label[5]], 2)
        # cv2.putText(image, label[5], (label[1], label[2]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[label[5]], 1)

    if display:
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


def non_maximum_supression(labels):

    remove_index = [0] * len(labels)
    labels = sorted(labels, key=lambda label: label[2]) 

    for i in range(0,len(labels) - 1):

        for j in range(i+1, len(labels)):

            box1 = labels[i]
            box2 = labels[j]

            value = nms_iou(box1[1],box1[2],box1[3],box1[4],box2[1],box2[2],box2[3],box2[4])
            
            if value >= NMS and box1[0] > box2[0]:
                remove_index[j] = 1
            elif value >= NMS and box1[0] <= box2[0]:
                remove_index[i] = 1
            else:
                pass

    new_labels = []
    for i in range(len(remove_index)):
        if remove_index[i] == 0:
            new_labels.append(labels[i])

    return new_labels



def nms_iou(fx,fy,fw,fh,px,py,pw,ph):
    # IOU
    intersect = (np.minimum(fx+fw, px+pw) - np.maximum(fx, px)) * (np.minimum(fy+fh, py+ph) - np.maximum(fy, py))
    union = (fw * fh) + (pw * ph) - intersect
    return intersect/union


