import cv2
import numpy as np
from test import load_model
import os
from generate_data import *

def resize_image(image, height, width, channels):

    o_height, o_width, _ = image.shape

    resized = np.zeros((height, width, channels))

    scale_factor = o_height / width

    resized_width = int(o_width/scale_factor)
    resized_height = int(o_height/scale_factor)

    res = cv2.resize(image, dsize=(resized_width,resized_height), interpolation=cv2.INTER_CUBIC)

    resized = res[0:height, 0:width, :]

    return resized/255.0



def main():

    directory = "models/"
    folders = [x[0] for x in os.walk(directory)]
    folders.sort()

    model = load_model(folders[-1])

    model.summary()

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture('demo_videos/13.MOV')

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            frame = resize_image(frame, 64, 64, 3)

            x_frame = np.expand_dims(frame, axis=0)
            # print("input.shape : "  + str(x_frame.shape))

            results = model.predict(x_frame)

            for r in range(len(results)):
                x_data = x_frame[r]
                y_data = results[r]

                image, labels = convert_data_to_image(x_data, y_data)
                rendered = render_with_labels(image, labels, display = False)
                # cv2.imwrite('output_tests/test_render_{:02d}.jpg'.format(r),rendered)


                # Display the resulting frame
                cv2.imshow('Frame',rendered/255.0)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
