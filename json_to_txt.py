import glob
import json
from common import *

def parse(folder_path):

    
    json_files = read_json(folder_path)

    with open("data/validation/validation.txt", "w+") as namefile:
        for f in json_files:
            output_name = f[:-5]
            namefile.write(output_name + ".jpg" + "\n")
    
    for file in json_files:

        boxes = [[0.0,0,0,0,0,0] for i in range(GRID_X * GRID_Y)]

        with open(file) as f:
            data = json.load(f)

            shapes = data["shapes"]

            for shape in shapes:
                
                label = shape["label"]

                point = shape["points"]

                x1 = point[0][0]
                y1 = point[0][1]
                x2 = point[1][0]
                y2 = point[1][1]

                x1 = (x1/64) * WIDTH
                x2 = (x2/64) * WIDTH
                y1 = (y1/64) * HEIGHT
                y2 = (y2/64) * HEIGHT

                width = x2 - x1
                height = y2 - y1

                scaled_width = width / WIDTH
                scaled_height = height / HEIGHT

                c_x = x1 + (width//2)
                c_y = y1 + (height//2)

                col = c_x // GRID_WIDTH
                row = c_y // GRID_HEIGHT

                x = c_x - (col * GRID_WIDTH)
                y = c_y - (row * GRID_HEIGHT)

                scaled_x = x/GRID_WIDTH
                scaled_y = y/GRID_HEIGHT

                boxes[int(row * GRID_X + col)] = [1.0, label, scaled_x, scaled_y, scaled_width, scaled_height]

            output_name = file[:-5]
            with open(output_name + ".txt", "w+") as txtfile:
                for box in boxes:
                    write_str = str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + " " + str(box[4]) + " " + str(box[5]) + "\n"
                    txtfile.write(write_str)
        

def read_json(folder_path):
    return glob.glob(folder_path + "/" + "*.json")



if __name__ == "__main__":
    # read_json("validation_frames")

    parse("test_data/handlabel/")

