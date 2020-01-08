
import tensorflow as tf
from tensorflow import keras
from train import *
from generate_data import *
import os

def load_model(model_path):

    with open(model_path + "/model.json") as json_file:
        json_config = json_file.read()

    model = keras.models.model_from_json(json_config)
    model.load_weights(model_path + "/model_weights.h5")

    return model

def main(model_path):

    model = load_model(model_path)

    x_tests, y_tests = next(generator(10, test=True))

    results = y_tests
    # results = model.predict(x_tests)
    # print(results)

    for r in range(len(results)):
        x_data = x_tests[r]
        y_data = results[r]

        image, labels = convert_data_to_image(x_data, y_data)
        rendered = render_with_labels(image, labels, display = True)
        cv2.imwrite('output_tests/test_render_{:02d}.jpg'.format(r),rendered)


if __name__ == "__main__":

    directory = "models/"
    folders = [x[0] for x in os.walk(directory)]
    folders.sort()
    print(folders[-1])

    main(folders[-1])
