from common import *
from utils import *

# Read the images
IMAGES, _ = read_subimages(MONEY_PATH)

def generate(output_currency = "RM50", angle = 0) :
    '''
    main function to generete the images
    @rows - number of rows for the image
    @col - number of columns for the image
    '''

    image = random.choice(IMAGES)

    # Generate the background
    background = generate_background()

    # Rotate the image
    rimage = rotate_image(image, angle)
    rotate_height, rotate_width, _ = rimage.shape

    # Calculating the height and width
    height, width, channels = background.shape

    height_of_note = int(math.floor(height * 0.8))
    width_of_note = int(math.floor(width * 0.8))

    default_box = {
        'confidence' : 0,
        'x': 0,
        'y': 0,
        'height': 0,
        'width': 0,
        'class' : 0
        }

    bounding_box_for_image = [default_box for i in range(GRID_X*GRID_Y)]

    # Calculate the x and y
    x_center = width * LOCATION_X
    y_center = height * LOCATION_Y

    # Resize the image
    if rotate_height > rotate_width:
        resize_image = image_resize(rimage, height=height_of_note)
    else:
        resize_image = image_resize(rimage, width=width_of_note)

    rheight, rwidth, rchannel = resize_image.shape
    
    # Calculate the top left x and y 
    x_top = abs(int(x_center - (rwidth // 2)))
    y_top = abs(int(y_center - (rheight // 2)))

    # Overlay the image to the background image
    final_image = overlay_transparent(background,resize_image,x_top,y_top)

    # # Update the brightness
    final_image = final_image * random.uniform(0.5, 1.5)

    final = np.clip(final_image,a_min = 0, a_max = 255.0)

    
    box = {
        'confidence': 1,
        'x': float(x_center - (x_center // GRID_WIDTH) * GRID_WIDTH ) / GRID_WIDTH,
        'y': float(y_center - (y_center // GRID_HEIGHT) * GRID_HEIGHT ) / GRID_HEIGHT,
        'height': float(rheight)/float(height),
        'width':float(rwidth)/float(width),
        'class' : str(CLASS[output_currency])
    }

    
    col_index = (x_center // GRID_WIDTH)
    row_index = (y_center // GRID_HEIGHT)
    loc = GRID_X * row_index + col_index
    bounding_box_for_image[int(loc)] = box

    # Create the label
    labels = np.zeros((GRID_Y, GRID_X, 5+len(CLASSES)))

    for row in range(GRID_X):
        for col in range(GRID_Y):
            labels[row, col, 0] = float(bounding_box_for_image[row * GRID_X + col]['confidence'])
            labels[row, col, 1:5] = [
                float(bounding_box_for_image[row * GRID_X + col]['x']),
                float(bounding_box_for_image[row * GRID_X + col]['y']),
                float(bounding_box_for_image[row * GRID_X + col]['width']),
                float(bounding_box_for_image[row * GRID_X + col]['height'])
            ]
            labels[row, col, int(5+float(bounding_box_for_image[row * GRID_X + col]['class']))] = float(1)

    return final_image,labels



def generator(batch_size):

    while True:
        # Empty batch arrays.
        x_trains = []
        y_trains = []
        # Create batch data.
        for i in range(batch_size):
            image, labels = generate(angle=random.randint(0,359))

            # Append
            x_trains.append(image)
            y_trains.append(labels)

        x_trains = np.asarray(x_trains).reshape((batch_size, HEIGHT, WIDTH, CHANNEL))
        y_trains = np.asarray(y_trains).reshape((batch_size, GRID_Y, GRID_X, 5+len(CLASSES)))
        yield x_trains, y_trains


if __name__ == "__main__" :
    # generate_multiple_angles()

    x_train,y_train = next(generator(8))

    for i in range(len(x_train)):
        x_data = x_train[i]
        y_data = y_train[i]

        image, label = convert_data_to_image(x_data, y_data)

        image = cv2.UMat(image).get()

        rendered = render_with_labels(image, label, display = False)

        cv2.imwrite('output_tests/test_render_{:02d}.png'.format(i),rendered)


