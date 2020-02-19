import cv2
import numpy as np
import random

import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

from common import *

def augmentation(aimage, apolygon):

    # Augmentation.
    seq = iaa.Sequential([
        # iaa.ChangeColorTemperature((4000, 9000)),
        # iaa.Rotate(rotate=(0, 359)),
        # iaa.Affine(scale=(0.8, 1.2)),
        # iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.5),
        # iaa.PerspectiveTransform(scale=(0.02, 0.08), keep_size=True),
        iaa.Resize({'width': WIDTH, 'height': HEIGHT}, interpolation=imgaug.ALL)
    ])

    count = len(apolygon)

    polygons = []
    for polygon in apolygon:
        polygons.append(Polygon(polygon['points']))
        polygon['points'] = []

    pps = PolygonsOnImage(polygons, shape=aimage.shape)

    # Augment.
    aimage, pps = seq(image=aimage, polygons=pps)

    for i,polygon in enumerate(apolygon):

        for (x,y) in pps[i]:

            if x <= 0:
                x = 0
            if y <= 0:
                y = 0
            if x >= WIDTH:
                x = WIDTH - 1
            if y >= HEIGHT:
                y = HEIGHT - 1

            polygon['points'].append([x,y])

    return aimage, apolygon

def generate_geometrical_noise(image):
    height, width, depth = image.shape

    # # Draw line.
    for _ in range(50):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        thickness = random.randint(1, 5)

        image = cv2.line(image, (x1,y1), (x2,y2), (r,g,b), thickness)

    # Draw rect.
    for _ in range(50):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        thickness = random.randint(1, 3)

        image = cv2.rectangle(image, (x1,y1), (x2,y2), (r,g,b), thickness)

    # Draw circle.
    for _ in range(200):
        x1 = random.randint(-width//2, width+width//2)
        y1 = random.randint(-height//2, height+height//2)
        radius = random.randint(0, width)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        thickness = random.randint(1, 3)

        image = cv2.circle(image, (x1,y1), radius, (r,g,b), thickness)

    return image

def change_brightness(image, mode = "uniform"):

    if mode == "uniform":
        image = image * random.uniform(0.5, 1.5)
        return np.clip(image,a_min = 0, a_max = 255.0)

    if mode == "transparent_triangle":

        pt1 = (random.randint(0,WIDTH), random.randint(0,HEIGHT))
        pt2 = (random.randint(0,WIDTH), random.randint(0,HEIGHT))
        pt3 = (random.randint(0,WIDTH), random.randint(0,HEIGHT))

        triangle_cnt = np.array( [pt1, pt2, pt3] )
        shape = cv2.drawContours(np.full((HEIGHT,WIDTH,CHANNEL), 255.), [triangle_cnt], 0, (0,0,0), -1)

        return cv2.addWeighted(shape,0.3,image,0.7,0)

def generate_background(mode = "noise"):

    if mode == "white" :

        return np.full((RHEIGHT,RWIDTH,CHANNEL), 255.)

    elif mode== "black" :

        return np.full((RHEIGHT,RWIDTH,CHANNEL), 0.)

    elif mode == "noise" :
        return np.random.randint(256.0, size=(RHEIGHT, RWIDTH,CHANNEL), dtype = np.uint8)

    elif mode == 'geometric':

        return generate_geometrical_noise(np.full((RHEIGHT,RWIDTH,CHANNEL), 1., dtype = np.uint8))

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


# def overlay_image(layer0_img, layer1_img, x, y):

#     height, width, channel = layer1_img.shape

#     layer0_img[y: y + height, x: x + width ] = layer1_img

#     return layer0_img

# def rotate_image(mat, angle):
#     """
#     Rotates an image (angle in degrees) and expands image to avoid cropping
#     """

#     height, width = mat.shape[:2] # image shape has 3 dimensions
#     image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

#     rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

#     # rotation calculates the cos and sin, taking absolutes of those.
#     abs_cos = abs(rotation_mat[0,0])
#     abs_sin = abs(rotation_mat[0,1])

#     # find the new width and height bounds
#     bound_w = int(height * abs_sin + width * abs_cos)
#     bound_h = int(height * abs_cos + width * abs_sin)

#     # subtract old image center (bringing image back to origo) and adding the new image center coordinates
#     rotation_mat[0, 2] += bound_w/2 - image_center[0]
#     rotation_mat[1, 2] += bound_h/2 - image_center[1]

#     # rotate image with the new bounds and translated rotation matrix
#     rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
#     return rotated_mat



# def doOverlap(first, second):
#     x_axis_not_overlap = False
#     y_axis_not_overlap = False

#     if(int(first["x1"]) > int(second["x2"]) or int(first["x2"]) < int(second["x1"])):
#         x_axis_not_overlap = True

#     if(int(first["y1"]) > int(second["y2"]) or int(first["y2"]) < int(second["y1"])):
#         y_axis_not_overlap = True

#     if x_axis_not_overlap and y_axis_not_overlap:
#         return False
#     else:
#         return True
