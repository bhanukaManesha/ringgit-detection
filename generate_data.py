#!/usr/bin/env python3
import random
import numpy as np
from common import *

def convert_image_to_data(image, texts):
    # Input.
    x_data = np.array(image.convert('L')) / 255
    x_data = np.reshape(x_data, [HEIGHT, WIDTH, CHANNEL])

    # Truth.
    y_data = np.zeros((GRID_Y, GRID_X, 5+len(TEXTS)))
    for t in texts:
        x, y ,w, h = t.x, t.y, t.w, t.h
        cx = (x + w/2) // GRID_WIDTH # CELL x
        cy = (y + h/2) // GRID_HEIGHT # CELL y
        bx = (x + w/2) % GRID_WIDTH / GRID_WIDTH # CENTER of box relative to box
        by = (y + h/2) % GRID_HEIGHT / GRID_HEIGHT # CENTER of box relative to box
        bw = w / WIDTH # WIDTH of box relative to image
        bh = h / HEIGHT # HEIGHT of box relative to image

        row, col = int(cy), int(cx) # Swap to row and col.
        y_data[row, col, 0] = 1
        y_data[row, col, 1:5] = [bx, by, bw, bh]
        y_data[row, col, 5+TEXTS.index(t.text)] = 1

    return x_data, y_data

def convert_data_to_image(x_data, y_data):
    # Input.
    x_data = np.reshape(x_data, [HEIGHT, WIDTH])
    image = Image.fromarray(np.uint8(x_data * 255))

    # Labels
    texts = []
    n_row, n_col, _ = y_data.shape
    for row in range(n_row):
        for col in range(n_col):
            d = y_data[row, col]
            # If text is available.
            if d[0] < 0.95:
                continue
            # Convert data.
            bx, by, bw, bh = d[1:5]
            w = bw * WIDTH
            h = bh * HEIGHT
            x = (col * GRID_WIDTH) + (bx * GRID_WIDTH) - w/2
            y = (row * GRID_HEIGHT) + (by * GRID_HEIGHT) - h/2
            s = TEXTS[np.argmax(d[5:])]
            # Text
            t = Text(s, x, y, w, h, 15)
            texts.append(t)

    return image, texts

def load_batch():
    pass

def main():
    image, texts = generate_image(WIDTH, HEIGHT, seeds=TEXTS)
    image.save('output_images/data_plain.png', 'PNG')

    # Convert image to data.
    x_data, y_data = convert_image_to_data(image, texts)
    # Convert data to image.
    image, texts = convert_data_to_image(x_data, y_data)

    # Save.
    rendered = render_with_texts(image, texts)
    rendered.save('output_images/data_render.png', 'PNG')



if __name__ == '__main__':
    main()