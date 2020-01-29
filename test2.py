
import tensorflow as tf
from tensorflow import keras
from train import *
from generate_data import *
import os
from train2 import *

def load_model(model_path):

    with open(model_path + "/model.json") as json_file:
        json_config = json_file.read()

    model = keras.models.model_from_json(json_config)
    model.load_weights(model_path + "/model_weights.h5")

    return model

def main(model_path):

    model = load_model(model_path)
    
    t_1_x_train,t_1_y_train = load_images_from_directory("test_data/t_train/")
    # t_back_x_train,t_back_y_train = load_images_from_directory("test_data/t_back/")
    t_real_x_train,t_real_y_train = load_images_from_directory("test_data/t_val/")

    # Append everything
    x_train = []
    y_train = []

    # x_train = np.concatenate((np.asarray(t_1_x_train),np.asarray(t_back_x_train)), axis=0)
    # y_train = np.concatenate((np.asarray(t_1_y_train),np.asarray(t_back_y_train)), axis=0)

    x_train = np.asarray(t_1_x_train)
    y_train = np.asarray(t_1_y_train)

    x_val = np.asarray(t_real_x_train)
    y_val = np.asarray(t_real_y_train)


    # ---------- Test

    # Remove the folder
    shutil.rmtree("output_tests/")
    
    # Create a folder
    directory = "output_tests"
    if not os.path.exists(directory):
        os.makedirs(directory)

        os.makedirs(directory + "/train")
        os.makedirs(directory + "/valid")
    

    # Plot training
    results = model.predict(x_train)

    for r in range(len(results)):
        x_data = x_train[r]
        y_data = results[r]

        image, texts = convert_data_to_image(x_data, y_data)
        rendered = render_with_labels(image, texts, display = False)
        cv2.imwrite('output_tests/train/test_render_{:02d}.png'.format(r),rendered)

    # Plot testing
    results = model.predict(x_val)

    for r in range(len(results)):
        x_data = x_val[r]
        y_data = results[r]

        image, texts = convert_data_to_image(x_data, y_data)
        rendered = render_with_labels(image, texts, display = False)
        cv2.imwrite('output_tests/valid/test_render_{:02d}.png'.format(r),rendered)



if __name__ == "__main__":

    directory = "models/"
    folders = [x[0] for x in os.walk(directory)]
    folders.sort()
    print(folders[-1])

    main(folders[-1])
