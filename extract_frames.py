import cv2
import numpy as np
from generate_data import *

def extract(save_path):

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture('demo_videos/15.MOV')

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    count = 0

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            
            small_frame = resize_image(frame, 64, 64, 3) * 255.0

            cv2.imwrite(save_path + "/" + str(count) + ".jpg",small_frame)

            count += 1
            

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    output_dir = "validation_frames"

    extract(output_dir)