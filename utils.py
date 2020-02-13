#!/usr/bin/env python3
from common import *
import tensorflow as tf
import cv2
import os
import pathlib
import numpy as np
import random
import math
import imutils
from uuid import uuid4
import json
import sys
import copy
import glob
import itertools

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

from common import *

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

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()

def read_subimages(path_to_folder):
    images = []
    files = glob.glob(path_to_folder + "*.png")

    for filename in files:
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)

    return images, files

def overlay_image(layer0_img, layer1_img, x, y):

    height, width, channel = layer1_img.shape

    layer0_img[y: y + height, x: x + width ] = layer1_img

    return layer0_img

def doOverlap(first, second):
    x_axis_not_overlap = False
    y_axis_not_overlap = False

    if(int(first["x1"]) > int(second["x2"]) or int(first["x2"]) < int(second["x1"])):
        x_axis_not_overlap = True

    if(int(first["y1"]) > int(second["y2"]) or int(first["y2"]) < int(second["y1"])):
        y_axis_not_overlap = True

    if x_axis_not_overlap and y_axis_not_overlap:
        return False
    else:
        return True

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

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

def convert_data_to_image(x_data, y_data):
    # Input.
    image = np.reshape(x_data, [HEIGHT, WIDTH, CHANNEL])
    image = np.uint8(x_data * 255)

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
            x = (col * GRID_WIDTH) + (bx * GRID_WIDTH) - w/2
            y = (row * GRID_HEIGHT) + (by * GRID_HEIGHT) - h/2

            s = CLASSES[np.argmax(d[5:])]

            # print([d[0],x,y,w,h,s])
            # labels
            labels.append([d[0],x,y,w,h,s])

    return image, labels

def load_image_names(mode):

    if mode == "train":
        filename = "data/train/train.txt"
    elif mode == "test":
        filename = "data/test/test.txt"
    elif mode == "valid":
        filename = "data/validation/validation.txt"

    with open(filename) as f:
        image_paths = f.readlines()
    image_paths = [x.strip() for x in image_paths]

    if mode == "train":
        image_paths = [x.replace("data/train/images/","") for x in image_paths]
    elif mode == "test":
        image_paths = [x.replace("data/test/images/","") for x in image_paths]
    elif mode == "valid":
        image_paths = [x.replace("data/validation/images/","") for x in image_paths]

    image_paths = [x.replace(".jpg","") for x in image_paths]

    return image_paths

def read_data(mode):

    if mode == "train":
        image_path = "data/train/images/"
    elif mode == "test":
        image_path = "data/test/images/"
    elif mode == "valid":
        image_path = "data/validation/images/"

    IMAGE_NAMES = load_image_names(mode=mode)
    print('total.images: ', len(IMAGE_NAMES))

    image_data = []
    annotation_data = []


    for image_name in IMAGE_NAMES:

        image_type = ".jpg"

        # print(image_path + image_name + image_type)
        image = cv2.imread(image_path + image_name + image_type).astype(np.float32)/255.0

        image_data.append(image)

        if mode == "train":
            label_path = "data/train/labels/"
        elif mode == "test":
            label_path = "data/test/labels/"
        elif mode == "valid":
            label_path = "data/validation/labels/"

        with open(label_path + image_name + ".txt") as f:
            annotations = f.readlines()


        annotations = [x.strip() for x in annotations]
        annotations = [x.split() for x in annotations]
        annotations = np.asarray(annotations)

        y_data = np.zeros((GRID_Y, GRID_X, 5+len(CLASSES)))

        for row in range(GRID_X):
            for col in range(GRID_Y):
                y_data[row, col, 0] = float(annotations[row * GRID_X + col][0])
                y_data[row, col, 1:5] = [
                    float(annotations[row * GRID_X + col][2]),
                    float(annotations[row * GRID_X + col][3]),
                    float(annotations[row * GRID_X + col][4]),
                    float(annotations[row * GRID_X + col][5])
                ]
                y_data[row, col, int(5+float(annotations[row * GRID_X + col][1]))] = float(1)

        annotation_data.append(y_data)

    image_data = np.asarray(image_data)
    annotation_data = np.asarray(annotation_data)

    return image_data, annotation_data

def load_image(images, labels):
    num = random.randrange(0, len(images))
    return images[num], labels[num]

def render_with_labels(image, labels, display):
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

def nms_iou(fx,fy,fw,fh,px,py,pw,ph):
    # IOU
    intersect = (np.minimum(fx+fw, px+pw) - np.maximum(fx, px)) * (np.minimum(fy+fh, py+ph) - np.maximum(fy, py))
    union = (fw * fh) + (pw * ph) - intersect
    return intersect/union

def load_images_from_directory(path):

    image_paths = glob.glob(path + "images/*.png")
    label_paths = glob.glob(path + "labels/*.json")

    image_paths.sort()
    label_paths.sort()

    x_train = []
    y_train = []

    for apath in glob.glob(path + '/images/*.png', recursive=True):
        aname = pathlib.Path(apath).stem
        image_path = '{}/images/{}.png'.format(path,aname)
        label_path = '{}/labels/{}.json'.format(path,aname)

        try:
            # Open label file.
            with open(label_path, 'r') as f:
                alabel = json.load(f)
            # Open image file.
            aimage = cv2.imread(image_path)
            assert(aimage.shape[0] == aimage.shape[1])

            aimage, alabel = augmentation(aimage, alabel)
            images, labels = convert_data_to_yolo(aimage, alabel)

            # Append to data.
            x_train.append(images)
            y_train.append(labels)
        except IOError as e:
            pass

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    return x_train,y_train

def convert_data_to_yolo(image, polygons):

    # Input.
    x_data = image / 255
    x_data = np.reshape(x_data, [HEIGHT, WIDTH, CHANNEL])

    # Truth.
    y_data = np.zeros((GRID_Y, GRID_X, 5+len(CLASSES)))

    for polygon in polygons:
        confidence = polygon['confidence']
        box_class = polygon['class']
        points = polygon['points']

        x_ = []
        y_ = []

        for point in points:
            x_.append(point[0])
            y_.append(point[1])

        x = min(x_)
        y = min(y_)
        w = max(x_) - x
        h = max(y_) - y

        cx = (x + w/2) // GRID_WIDTH # CELL x
        cy = (y + h/2) // GRID_HEIGHT # CELL y
        bx = (x + w/2) % GRID_WIDTH / GRID_WIDTH # CENTER of box relative to box
        by = (y + h/2) % GRID_HEIGHT / GRID_HEIGHT # CENTER of box relative to box
        bw = w / WIDTH # WIDTH of box relative to image
        bh = h / HEIGHT # HEIGHT of box relative to image

        row, col = int(cy), int(cx) # Swap to row and col.
        y_data[row, col, 0] = 1
        y_data[row, col, 1:5] = [bx, by, bw, bh]
        y_data[row, col, 5+box_class] = 1.0


    return x_data, y_data

def augmentation(aimage, apolygons):
    # Augmentation.
    seq = iaa.Sequential([
        iaa.Rotate(rotate=(0, 360)),
        iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.5),
        iaa.PerspectiveTransform(scale=(0.01, 0.09), keep_size=True),
        iaa.Resize({'width': WIDTH, 'height': HEIGHT}, interpolation=cv2.INTER_AREA)
    ])

    count = len(apolygons)

    polygons = []
    for polygon in apolygons:
        polygons.append(Polygon(polygon['points']))


    for _ in range(count):
        pps = PolygonsOnImage(polygons, shape=aimage.shape)

        # Augment.
        aimage, pps = seq(image=aimage, polygons=pps)

        # print(pps)
        # print(aimage.shape)

    for i,polygon in enumerate(apolygons):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pps[i]

        if (x1 < 0) :
            x1 = 0
        if (x2 < 0) :
            x2 = 0
        if (x3 < 0) :
            x3 = 0
        if (x4 < 0) :
            x4 = 0

        if (x1 >= WIDTH):
            x1 = WIDTH
        if (x2 >= WIDTH):
            x2 = WIDTH
        if (x3 >= WIDTH):
            x3 = WIDTH
        if (x4 >= WIDTH):
            x4 = WIDTH

        if (y1 < 0) :
            y1 = 0
        if (y2 < 0) :
            y2 = 0
        if (y3 < 0) :
            y3 = 0
        if (y4 < 0) :
            y4 = 0

        if (y1 >= HEIGHT):
            y1 = HEIGHT
        if (y2 >= HEIGHT):
            y2 = HEIGHT
        if (y3 >= HEIGHT):
            y3 = HEIGHT
        if (y4 >= HEIGHT):
            y4 = HEIGHT


        polygon['points'] = [
            [x1,y1],[x2,y2],[x3,y3],[x4,y4]
        ]



    return aimage, apolygons








if __name__ == "__main__":
    x_train, y_train = load_images_from_directory("data/val/")
    print(x_train.shape)
    print(y_train.shape)
    for i in range(len(x_train)):
        x_data = x_train[i]
        y_data = y_train[i]

        image, label = convert_data_to_image(x_data, y_data)
        rendered = render_with_labels(image, label, display = False)
        cv2.imwrite('output_render/test_render_{:02d}.png'.format(i),rendered)