#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras
from train import *
import os

def load_model(model_path):

    with open(model_path + "/model.json") as json_file:
        json_config = json_file.read()

    model = keras.models.model_from_json(json_config)
    model.load_weights(model_path + "/model_weights.h5")

    return model

def main(model_path):

    model = load_model(model_path)

    x_train_2,y_train_2 = next(generator(10))
    x_test,_ = load_images_from_directory(validation_path)
    x_test = np.concatenate((np.asarray(x_train_2),np.asarray(x_test)),axis=0)

    # Remove the folder
    shutil.rmtree("output_tests/")

    # Create a folder
    directory = "output_tests"
    if not os.path.exists(directory):
        os.makedirs(directory)


    results = model.predict(x_test)

    # Plot training
    for r in range(len(results)):
        x_data = x_test[r]
        y_data = results[r]

        image, labels = convert_data_to_image(x_data, y_data)
        labels = non_maximum_supression(labels)
        rendered = render_with_labels(image, labels, display = False)
        cv2.imwrite('output_tests/test_render_{:02d}.png'.format(r),rendered)


if __name__ == "__main__":

    directory = "models/"
    folders = [x[0] for x in os.walk(directory)]
    folders.sort()
    print(folders[-1])

    main(folders[-1])
