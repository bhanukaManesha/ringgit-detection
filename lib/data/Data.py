
import numpy as np
from common import *

class Data:
    def __init__(self, data, dtype):
        self.x, self.y = data

        # Can be : data, render, batch_data
        self.dtype = dtype

    def asrender(self):
        # Input.
        self.x = np.reshape(self.x, [HEIGHT, WIDTH, CHANNEL])
        self.x = np.uint8(self.x * 255)

        # Labels
        labels = []

        n_row, n_col, _ = self.y.shape

        for row in range(n_row):
            for col in range(n_col):

                d = self.y[row, col]
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

        self.y = labels
        self.type = 'render'

    def asdata(self):

        # Input.
        self.x = self.x / 255
        self.x = np.reshape(self.x, [HEIGHT, WIDTH, CHANNEL])

        # Truth.
        y_data = np.zeros((GRID_Y, GRID_X, 5+len(CLASSES)))

        for polygon in self.y:
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

        self.y = y_data
        self.type = 'data'
