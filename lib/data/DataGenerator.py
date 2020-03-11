from tqdm import tqdm
from copy import deepcopy
import math, os, random, glob, pathlib, cv2, json
import numpy as np

from common import *

from lib.data.Data import Data
from lib.data.AugmentData import AugmentData

class DataGenerator:

    def __init__(self):
        self.images = []
        self.polygons = []

        self.background_generator = self._get_real_background()

        for aclass in CLASSES:
            x_, y_ = self.read_polygons('{}/{}'.format('data/notes',aclass))
            self.images.append(x_)
            self.polygons.append(y_)

    # Can Optimize
    def from_raw(self, x_images,y_polygons,no_images = 1):
        '''
        main function to generete the images
        @rows - number of rows for the image
        @col - number of columns for the image
        '''
        allpolygons = []

        # Generate the background
        background = next(self.background_generator)

        for _ in range(no_images):

            output_currency = random.choice(CLASSES)
            output_currency_index = CLASS[output_currency]

            # Get random image and respective set of points
            rand_int = random.randint(0,len(x_images[output_currency_index]) - 1)

            image = x_images[CLASS[output_currency]][rand_int]
            points = np.asarray(y_polygons[CLASS[output_currency]][rand_int])

            rotate_height, rotate_width, _ = image.shape

            # Calculating the height and width
            height, width, channels = background.shape

            random_size = random.uniform(0.8, 0.95)
            height_of_note = int(math.floor(height * random_size))
            width_of_note = int(math.floor(width * random_size))

            # Calculate the x and y
            x_center = width * random.uniform(0.3, 0.7)
            y_center = height * random.uniform(0.3, 0.7)

            # Resize the image
            if rotate_height > rotate_width:
                resize_image = self._image_resize(image, height=height_of_note)
            else:
                resize_image = self._image_resize(image, width=width_of_note)

            rheight, rwidth, rchannel = resize_image.shape
            oheight, owidth, _ = image.shape

            # Calculate the top left x and y
            x_top = abs(int(x_center - (rwidth // 2)))
            y_top = abs(int(y_center - (rheight // 2)))

            resizer = lambda t: [x_top + ((t[0] / owidth) * rwidth) , y_top + ((t[1] / oheight) * rheight)]
            points = np.apply_along_axis(resizer, 1, points)

            # Overlay the image to the background image
            final_image = self._overlay_transparent(background,resize_image,x_top,y_top)

            polygon = {
                'confidence' : 1.0,
                'points':points,
                'class' : CLASS[output_currency]
                }

            allpolygons.append(polygon)

        return Data(data = (final_image, allpolygons), dtype='render')

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

        return np.asarray(images), np.asarray(points)

    def serve(self,batch_size):

        datas = []

        while True:
            # Create batch data.
            for i in tqdm(range(batch_size)):

                aug = AugmentData.fromdataobj(self.from_raw(deepcopy(self.images), deepcopy(self.polygons)))

                aug.augmentation()

                aug.asdata()

                # Append
                datas.append(aug)

            yield self.merge_data_objs(datas, batch_size)

    def merge_data_objs(self,datas, batch_size):

        x_ = []
        y_ = []

        for data in datas:
            x_.append(data.x)
            y_.append(data.y)

        return Data(
            data = (np.asarray(x_).reshape((batch_size, HEIGHT, WIDTH, CHANNEL)), np.asarray(y_).reshape((batch_size, GRID_Y, GRID_X, 5+len(CLASSES)))),
            dtype = 'batch_data'
        )

    def _get_real_background(self):

        bimages = self._images_from_directory('data/raw_backgrounds')

        print("{} background images.".format(len(bimages)))

        while True:

            image = random.choice(bimages)

            image = self._image_resize(image, width = RWIDTH, height=RHEIGHT)

            yield image

    def _images_from_directory(self, folder):

        images = []

        image_paths = glob.glob("{}/*.jpeg".format(folder))


        for path in image_paths:

            images.append(cv2.imread(path))

        return images

    def generate_geometrical_noise(self,image):
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

    def _adjust_brightness(self, image, mode = "uniform"):

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

    def _generate_background(self, mode = "real"):

        if mode == "white" :

            return np.full((RHEIGHT,RWIDTH,CHANNEL), 255.)

        elif mode== "black" :

            return np.full((RHEIGHT,RWIDTH,CHANNEL), 0.)

        elif mode == "noise" :
            return np.random.randint(256.0, size=(RHEIGHT, RWIDTH,CHANNEL), dtype = np.uint8)

        elif mode == 'geometric':

            return generate_geometrical_noise(np.full((RHEIGHT,RWIDTH,CHANNEL), 1., dtype = np.uint8))

    def _image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
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

    def _overlay_transparent(self, background, overlay, x, y):

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

    def update_label_class(label):
        label["class"] = CLASS[label["class"]]
        return label



    def from_directory(self, folder):

        image_paths = glob.glob("{}/images/*.png".format(folder))
        label_paths = glob.glob("{}/labels/*.json".format(folder))

        image_paths.sort()
        label_paths.sort()

        datas = []

        for apath in glob.glob('{}/images/*.png'.format(folder), recursive=True):
            aname = pathlib.Path(apath).stem
            image_path = '{}/images/{}.png'.format(folder,aname)
            label_path = '{}/labels/{}.json'.format(folder,aname)

            try:
                # Open label file.
                with open(label_path, 'r') as f:
                    alabel = json.load(f)
                alabel = update_label_class(alabel)
                
                # Open image file.
                aimage = cv2.imread(image_path)
                assert(aimage.shape[0] == aimage.shape[1])

                aug = AugmentData((aimage,alabel), dtype='data')

                aug.augmentation()

                aug.asdata()

                # Append to data.
                datas.append(aug)

            except IOError as e:
                print(e)

        return self.merge_data_objs(datas, len(image_paths))
