#!/usr/bin/env python
from common import *
import glob
import cv2
import numpy as np
import os

def resize_image(image, height, width, channels):

    o_height, o_width, _ = image.shape

    resized = np.zeros((height, width, channels))

    scale_factor = o_height / width

    resized_width = int(o_width/scale_factor)
    resized_height = int(o_height/scale_factor)

    res = cv2.resize(image, dsize=(resized_width,resized_height), interpolation=cv2.INTER_CUBIC)

    resized = res[0:height, 0:width, :]

    return resized/255.0


def resize_images_in_folder(folder_path):

    image_paths = glob.glob(folder_path + "/*.jpg")

    for path in image_paths:
        
        image = cv2.imread(path)

        new_image = resize_image(image, WIDTH, HEIGHT, CHANNEL) * 255.0

        # Create a folder
        directory = folder_path + "resized"
        if not os.path.exists(directory):
            os.makedirs(directory)

        image_name = path.replace(folder_path, "")

        cv2.imwrite(folder_path + "resized/" + image_name, new_image)

    

if __name__ == "__main__":
    resize_images_in_folder("test_data/old/random_b/images/")

