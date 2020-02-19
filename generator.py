#!/usr/bin/env python3
from argparse import ArgumentParser
from tqdm import tqdm
from copy import deepcopy
import math, os, random
import numpy as np


from common import *
import aug
from data import Data
from render import Render

class Generator:

    def generate_polygon_from_raw(self, x_images,y_polygons,no_images = 1):
        '''
        main function to generete the images
        @rows - number of rows for the image
        @col - number of columns for the image
        '''
        allpolygons = []

        # Generate the background
        background = aug.generate_background()

        for _ in range(no_images):

            output_currency = random.choice(CLASSES)
            output_currency_index = CLASS[output_currency]

            # Get random image and respective set of points
            rand_int = random.randint(0,len(x_images[output_currency_index]) - 1)

            image = x_images[CLASS[output_currency]][rand_int]
            points = y_polygons[CLASS[output_currency]][rand_int]

            rotate_height, rotate_width, _ = image.shape

            # Calculating the height and width
            height, width, channels = background.shape

            random_size = random.uniform(0.6, 0.75)
            height_of_note = int(math.floor(height * random_size))
            width_of_note = int(math.floor(width * random_size))

            # Calculate the x and y
            x_center = width * random.uniform(0.3, 0.7)
            y_center = height * random.uniform(0.3, 0.7)

            # Resize the image
            if rotate_height > rotate_width:
                resize_image = aug.image_resize(image, height=height_of_note)
            else:
                resize_image = aug.image_resize(image, width=width_of_note)

            rheight, rwidth, rchannel = resize_image.shape
            oheight, owidth, _ = image.shape

            # Calculate the top left x and y
            x_top = abs(int(x_center - (rwidth // 2)))
            y_top = abs(int(y_center - (rheight // 2)))

            for i, [x,y] in enumerate(points):
                nx = (x / owidth) * rwidth
                ny = (y / oheight) * rheight

                points[i] = [nx + x_top,ny + y_top]

            # Overlay the image to the background image
            final_image = aug.overlay_transparent(background,resize_image,x_top,y_top)

            polygon = {
                'confidence' : 1.0,
                'points':points,
                'class' : CLASS[output_currency]
                }

            allpolygons.append(polygon)

        return final_image, allpolygons

    def serve(self,batch_size):

        x_images = []
        y_polygons = []

        for aclass in CLASSES:

            # Fix this asap
            data = Data()

            x, y = data.read_polygons('{}/{}'.format('data/raw_notes',aclass))
            x_images.append(x)
            y_polygons.append(y)

        while True:
            # Empty batch arrays.
            x_trains = []
            y_trains = []
            # Create batch data.
            for i in tqdm(range(batch_size)):

                image, polygons = self.generate_polygon_from_raw(deepcopy(x_images), deepcopy(y_polygons))
                image , polygons = aug.augmentation(image, polygons)
                image, labels = data.convert_render_to_data(image, polygons)

                # Append
                x_trains.append(image)
                y_trains.append(labels)

            x_trains = np.asarray(x_trains).reshape((batch_size, HEIGHT, WIDTH, CHANNEL))
            y_trains = np.asarray(y_trains).reshape((batch_size, GRID_Y, GRID_X, 5+len(CLASSES)))
            yield x_trains, y_trains


if __name__ == "__main__" :
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("-c", "-count", dest="count",
                        help="number of training data", metavar="file")
    parser.add_argument("-r", "-render", dest="render", default= False,
                        help="render each image")
    args = parser.parse_args()

    if args.count == None and int(args.count) > 50:
        print("-c enter a number greater than 50")

    render = True if args.render else False

    COUNT = int(args.count)

    # Create a generator
    generator = Generator()
    x_train,y_train = next(generator.serve(COUNT))

    # Fix this asap
    data = Data()
    data.write_pickle_datas(datas=(x_train, y_train))

    if render :
        r = Render()
        r.output_result(x_train, y_train, folder= 'output_render')
