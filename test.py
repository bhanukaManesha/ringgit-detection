
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

    x_test, y_test = next(generator(32))

    real_x_train,real_y_train = load_images_from_directory("real/")
    x_val = np.concatenate((np.asarray(real_x_train),np.asarray(x_test)), axis=0)
    y_val = np.concatenate((np.asarray(real_y_train),np.asarray(y_test)), axis=0)


    # Remove the folder
    shutil.rmtree("output_tests/")
    
    # Create a folder
    directory = "output_tests"
    if not os.path.exists(directory):
        os.makedirs(directory)
    

    results = model.predict(x_test)

    # Plot training
    for r in range(len(results)):
        x_data = x_val[r]
        y_data = results[r]
        # y_data = y_val[r]

        image, texts = convert_data_to_image(x_data, y_data)
        rendered = render_with_labels(image, texts, display = False)
        cv2.imwrite('output_tests/test_render_{:02d}.png'.format(r),rendered)


if __name__ == "__main__":

    directory = "models/"
    folders = [x[0] for x in os.walk(directory)]
    folders.sort()
    print(folders[-1])

    main(folders[-1])
