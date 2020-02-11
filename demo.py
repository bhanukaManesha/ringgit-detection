#!/usr/bin/env python
import cv2
import numpy as np
import os
from argparse import ArgumentParser
from utils import *

def main():

    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename",
                        help="path to the file", metavar="file")
    args = parser.parse_args()

    # load the model
    try:
        directory = "models/"
        folders = [x[0] for x in os.walk(directory)]
        folders.sort()
        model = load_model(folders[-1])
        model.summary()
        
    except:
        print("Please check the model.")
        return


    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(args.filename)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output/output.mp4',fourcc, 20.0, (1080,1080))

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            ori_frame = frame[:1080,:1080,:]
            ori_x_frame = np.expand_dims(ori_frame, axis=0)

            small_frame = image_resize(ori_frame, width=64, height=64)

            x_frame = np.expand_dims(small_frame, axis=0)

            results = model.predict(x_frame)

            for r in range(len(results)):
                x_data = ori_x_frame[r]
                y_data = results[r]

                
                image, labels = convert_data_to_image(x_data, y_data)
                labels = non_maximum_supression(labels)
                rendered = render_with_labels(image, labels, display = False)
                # cv2.imwrite('output_tests/test_render_{:02d}.jpg'.format(r),rendered)

                # rendered = np.uint8(rendered)

                # # Display the resulting frame
                cv2.imshow('Frame',rendered)
                
                # write the output frame to file
                out.write(rendered)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
