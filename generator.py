#!/usr/bin/env python3
from common import *
from utils import *
from argparse import ArgumentParser
from tqdm import tqdm
from copy import deepcopy

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

def generate_background(mode = "geometric"):

    if mode == "white" :

        return np.full((RHEIGHT,RWIDTH,CHANNEL), 255.)

    elif mode== "black" :

        return np.full((RHEIGHT,RWIDTH,CHANNEL), 0.)

    elif mode == "noise" :
        return np.random.randint(256.0, size=(RHEIGHT, RWIDTH,CHANNEL))

    elif mode == 'geometric':

        return generate_geometrical_noise(np.full((RHEIGHT,RWIDTH,CHANNEL), 1., dtype = np.uint8))

def generate_polygon(x_images,y_polygons,no_images = 1):
    '''
    main function to generete the images
    @rows - number of rows for the image
    @col - number of columns for the image
    '''

    allpolygons = []

    # Generate the background
    background = generate_background()

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
            resize_image = image_resize(image, height=height_of_note)
        else:
            resize_image = image_resize(image, width=width_of_note)

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
        final_image = overlay_transparent(background,resize_image,x_top,y_top)

        polygon = {
            'confidence' : 1.0,
            'points':points,
            'class' : CLASS[output_currency]
            }

        allpolygons.append(polygon)

    return final_image, allpolygons

def generate(images) :
    '''
    main function to generete the images
    @rows - number of rows for the image
    @col - number of columns for the image
    '''

    output_currency=random.choice(CLASSES)

    image = random.choice(images[CLASS[output_currency]])

    # Generate the background
    background = generate_background()

    # Rotate the image
    # rimage = rotate_image(image, angle)
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
        resize_image = image_resize(image, height=height_of_note)
    else:
        resize_image = image_resize(image, width=width_of_note)

    rheight, rwidth, rchannel = resize_image.shape

    # Calculate the top left x and y
    x_top = abs(int(x_center - (rwidth // 2)))
    y_top = abs(int(y_center - (rheight // 2)))

    # Overlay the image to the background image
    final_image = overlay_transparent(background,resize_image,x_top,y_top)

    # Change Brightness
    # final_image = change_brightness(final_image)

    polygon = {
        'confidence' : 1.0,
        'points':[
            [x_top,y_top],
            [x_top + rwidth, y_top],
            [x_top + rwidth, y_top + rheight],
            [x_top, y_top + rheight]
        ],
        'class' : CLASS[output_currency]
        }

    return final_image, [polygon]

def generator(batch_size):

    x_images = []
    y_polygons = []

    for aclass in CLASSES:
        x, y = read_polygons('{}/{}'.format(money_path,aclass))
        x_images.append(x)
        y_polygons.append(y)


    while True:
        # Empty batch arrays.
        x_trains = []
        y_trains = []
        # Create batch data.
        for i in tqdm(range(batch_size)):

            image, polygons = generate_polygon(deepcopy(x_images), deepcopy(y_polygons))
            image , polygons = augmentation(image, polygons)
            image, labels = convert_data_to_yolo(image, polygons)

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
    parser.add_argument("-r", "-render", dest="render",
                        help="render each image")
    args = parser.parse_args()

    if args.count == None and int(args.count) > 50:
        print("-c enter a number greater than 50")

    if args.render == None:
        render = False
    else:
        render = True

    COUNT = int(args.count)

    x_train,y_train = next(generator(COUNT))

    # Create a folder
    directory = "pickles"
    if not os.path.exists(directory):
        os.makedirs(directory)

    write_pickle_datas('pickles/trains.pickle', datas=(x_train, y_train))

    if render :
        for i in range(50):
            x_data = x_train[i]
            y_data = y_train[i]

            image, label = convert_data_to_image(x_data, y_data)
            rendered = render_with_labels(image, label, display = False)
            cv2.imwrite('output_render/test_render_{:02d}.png'.format(i),rendered)
