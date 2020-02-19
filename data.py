import glob, pathlib, cv2, json, pickle
import numpy as np
import os

import aug
from common import *



class Data:

    def __init__(self):
        self.val_folder = 'data/val'
        self.train_pickle_folder = 'data/pickles'
        self.train_pickle_name = 'trains.pickle'

    def read_polygons(self, folder):
        images = []
        points = []

        for apath in sorted(glob.glob('{}/images/*.png'.format(folder))):

            aname = pathlib.Path(apath).stem
            image_path = '{}/images/{}.png'.format(folder, aname)
            label_path = '{}/labels/{}.json'.format(folder, aname)

            try:
                # Open image file.
                aimage = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

                # Open label file.
                with open(label_path, 'r') as f:
                    alabel = json.load(f)

                # Append to data.
                images.append(aimage)
                points.append(alabel['points'])

                # print(apath)
            except IOError as e:
                pass

        return images, points



    def load_images_from_directory(self):

        image_paths = glob.glob("{}/images/*.png".format(self.val_folder))
        label_paths = glob.glob("{}/labels/*.json".format(self.val_folder))

        image_paths.sort()
        label_paths.sort()

        x_train = []
        y_train = []

        for apath in glob.glob('{}/images/*.png'.format(self.val_folder), recursive=True):
            aname = pathlib.Path(apath).stem
            image_path = '{}/images/{}.png'.format(self.val_folder,aname)
            label_path = '{}/labels/{}.json'.format(self.val_folder,aname)

            try:
                # Open label file.
                with open(label_path, 'r') as f:
                    alabel = json.load(f)
                # Open image file.
                aimage = cv2.imread(image_path)
                assert(aimage.shape[0] == aimage.shape[1])

                aimage, alabel = aug.augmentation(aimage, alabel)
                images, labels = self.convert_render_to_data(aimage, alabel)

                # Append to data.
                x_train.append(images)
                y_train.append(labels)
            except IOError as e:
                pass

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        return x_train,y_train

    def convert_data_to_render(self, x_data, y_data):
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


    def convert_render_to_data(self, image, polygons):

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

    def write_pickle_datas(self, datas):

        # Create a folder
        if not os.path.exists(self.train_pickle_folder):
            os.makedirs(self.train_pickle_folder)

        with open('{}/{}'.format(self.train_pickle_folder,self.train_pickle_name), 'wb') as f:
            pickle.dump(datas, f)

    def read_pickle_datas(self):
        with open('{}/{}'.format(self.train_pickle_folder, self.train_pickle_name), 'rb') as f:
            return pickle.load(f)