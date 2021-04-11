import pickle
from os import listdir

import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer


def save_data(raw_folder='data/'):
    dest_size = (26, 45)
    print("Bắt đầu xử lý ảnh...")
    pixels = []
    labels = []
    # Lặp qua các folder con trong thư mục raw
    for folder in listdir(raw_folder):
        if folder != '.DS_Store':
            print("Folder=", folder)
            # Lặp qua các file trong từng thư mục chứa các em
            for file in listdir(raw_folder + folder):
                if file != '.DS_Store':
                    print("File=", file)
                    pixels.append(cv2.resize(cv2.imread(raw_folder + folder + "/" + file), dsize=dest_size))
                    labels.append(folder)

    pixels = np.array(pixels)
    labels = np.array(labels)  # .reshape(-1,1)

    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    print(labels)

    file = open('pix.data', 'wb')
    # dump information to that file
    pickle.dump((pixels, labels), file)
    # close the file
    file.close()
    return


save_data()
