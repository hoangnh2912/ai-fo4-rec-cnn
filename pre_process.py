import cv2
import numpy as np


def process_img(path):
    img = cv2.imread(path, 0)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    edges = cv2.Canny(img, 100, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for idx, con in enumerate(contours):
        x, y, w, h = cv2.boundingRect(con)
        if 40 < w * h < 2000:
            cv2.drawContours(blank_image, contours, idx, (0, 255, 0), 1)
            if (w * h) > 260:
                cv2.imwrite("cache/" + str(idx) + ".png", blank_image[y:y + h, x:x + w])


# rename về image_0...
def rename_files():
    import os
    path = 'data_pre'
    files = os.listdir(path)
    for index, file in enumerate(files):
        os.rename(os.path.join(path, file), os.path.join(path, ''.join(['image_', str(index), '.jpg'])))


# rename_files()
process_img('data_pre/image_40.jpg')

