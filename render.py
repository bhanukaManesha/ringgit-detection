#!/usr/bin/env python3
import numpy as np
from uuid import uuid4
import cv2, shutil, os
import tensorflow as tf

from common import *
from aug import augmentation
from data import Data

class Render:

    def non_maximum_supression(self, labels):

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

    def render_with_labels(self, image, labels, display = False):

        for label in labels:
            # cv2.rectangle(image,(0,0),(5,5),(0,255,0),2)
            cv2.rectangle(image, (int(label[1]),int(label[2])), (int(label[1]+label[3]),int(label[2]+label[4])), colors[label[5]], 2)
            # cv2.putText(image, label[5], (label[1], label[2]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[label[5]], 1)

        if display:
            cv2.imshow('image',image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image

    def output_result(self, x, results, folder = "output_tests"):

        try:
            # Remove the folder
            shutil.rmtree("{}/".format(folder))

        except FileNotFoundError:
            pass

        # Create a folder
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Plot training
        for r in range(len(results)):
            x_data = x[r]
            y_data = results[r]

            # Fix this asap
            data = Data()

            image, labels = data.convert_data_to_render(x_data, y_data)
            labels = self.non_maximum_supression(labels)
            fimage = self.render_with_labels(image, labels)
            cv2.imwrite('{}/test_render_{:02d}.png'.format(folder,r),fimage)

# def read_images(path_to_folder):
#     images = []
#     files = glob.glob(path_to_folder + "*.png")

#     for filename in files:
#         img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
#         if img is not None:
#             images.append(img)

#     return images, files


# def load_image_names(mode):

#     if mode == "train":
#         filename = "data/train/train.txt"
#     elif mode == "test":
#         filename = "data/test/test.txt"
#     elif mode == "valid":
#         filename = "data/validation/validation.txt"

#     with open(filename) as f:
#         image_paths = f.readlines()
#     image_paths = [x.strip() for x in image_paths]

#     if mode == "train":
#         image_paths = [x.replace("data/train/images/","") for x in image_paths]
#     elif mode == "test":
#         image_paths = [x.replace("data/test/images/","") for x in image_paths]
#     elif mode == "valid":
#         image_paths = [x.replace("data/validation/images/","") for x in image_paths]

#     image_paths = [x.replace(".jpg","") for x in image_paths]

#     return image_paths

# def read_data(mode):

#     if mode == "train":
#         image_path = "data/train/images/"
#     elif mode == "test":
#         image_path = "data/test/images/"
#     elif mode == "valid":
#         image_path = "data/validation/images/"

#     IMAGE_NAMES = load_image_names(mode=mode)
#     print('total.images: ', len(IMAGE_NAMES))

#     image_data = []
#     annotation_data = []


#     for image_name in IMAGE_NAMES:

#         image_type = ".jpg"

#         # print(image_path + image_name + image_type)
#         image = cv2.imread(image_path + image_name + image_type).astype(np.float32)/255.0

#         image_data.append(image)

#         if mode == "train":
#             label_path = "data/train/labels/"
#         elif mode == "test":
#             label_path = "data/test/labels/"
#         elif mode == "valid":
#             label_path = "data/validation/labels/"

#         with open(label_path + image_name + ".txt") as f:
#             annotations = f.readlines()


#         annotations = [x.strip() for x in annotations]
#         annotations = [x.split() for x in annotations]
#         annotations = np.asarray(annotations)

#         y_data = np.zeros((GRID_Y, GRID_X, 5+len(CLASSES)))

#         for row in range(GRID_X):
#             for col in range(GRID_Y):
#                 y_data[row, col, 0] = float(annotations[row * GRID_X + col][0])
#                 y_data[row, col, 1:5] = [
#                     float(annotations[row * GRID_X + col][2]),
#                     float(annotations[row * GRID_X + col][3]),
#                     float(annotations[row * GRID_X + col][4]),
#                     float(annotations[row * GRID_X + col][5])
#                 ]
#                 y_data[row, col, int(5+float(annotations[row * GRID_X + col][1]))] = float(1)

#         annotation_data.append(y_data)

#     image_data = np.asarray(image_data)
#     annotation_data = np.asarray(annotation_data)

#     return image_data, annotation_data

# def load_image(images, labels):
#     num = random.randrange(0, len(images))
#     return images[num], labels[num]


    # def generate_image_from_image(self, images) :
    #     '''
    #     main function to generete the images
    #     @rows - number of rows for the image
    #     @col - number of columns for the image
    #     '''

    #     output_currency=random.choice(CLASSES)
    #     image = random.choice(images[CLASS[output_currency]])

    #     # Generate the background
    #     background = generate_background()

    #     # Rotate the image
    #     # rimage = rotate_image(image, angle)
    #     rotate_height, rotate_width, _ = image.shape

    #     # Calculating the height and width
    #     height, width, channels = background.shape

    #     random_size = random.uniform(0.6, 0.75)
    #     height_of_note = int(math.floor(height * random_size))
    #     width_of_note = int(math.floor(width * random_size))

    #     # Calculate the x and y
    #     x_center = width * random.uniform(0.3, 0.7)
    #     y_center = height * random.uniform(0.3, 0.7)

    #     # Resize the image
    #     if rotate_height > rotate_width:
    #         resize_image = image_resize(image, height=height_of_note)
    #     else:
    #         resize_image = image_resize(image, width=width_of_note)

    #     rheight, rwidth, rchannel = resize_image.shape

    #     # Calculate the top left x and y
    #     x_top = abs(int(x_center - (rwidth // 2)))
    #     y_top = abs(int(y_center - (rheight // 2)))

    #     # Overlay the image to the background image
    #     final_image = overlay_transparent(background,resize_image,x_top,y_top)

    #     # Change Brightness
    #     # final_image = change_brightness(final_image)

    #     polygon = {
    #         'confidence' : 1.0,
    #         'points':[
    #             [x_top,y_top],
    #             [x_top + rwidth, y_top],
    #             [x_top + rwidth, y_top + rheight],
    #             [x_top, y_top + rheight]
    #         ],
    #         'class' : CLASS[output_currency]
    #         }

    #     return final_image, [polygon]
