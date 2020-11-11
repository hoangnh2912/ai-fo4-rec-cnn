from time import time

import cv2
import numpy as np

shape = (26, 45)
class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'none']
padding_size = 50
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (0, 255, 0)
thickness = 1


def center_box(box):
    _, _, x, y, w, h = box
    return x + w / 2, y + h / 2


def euclid(box1, box2):
    x1, y1 = center_box(box1)
    x2, y2 = center_box(box2)
    return np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))


def not_in_list_predict(box, list_predict):
    _ = ""
    x1, y1, w1, h1 = box
    current = _, _, x1, y1, w1, h1
    cx1, cy1 = center_box(current)
    for other_pre in list_predict:
        _, _, _, _, w2, h2 = other_pre
        cx2, cy2 = center_box(other_pre)
        if cx1 == cx2 and w2 * h2 >= w1 * h1:
            return False
    return True


def using(path, my_model):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (512, 248))
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    edges = cv2.Canny(img, 100, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    process_con = []

    start = time()
    for idx, con in enumerate(contours):
        box = cv2.boundingRect(con)
        x, y, w, h = box
        if 40 < w * h < 2000:
            cv2.drawContours(blank_image, contours, idx, (0, 255, 0), 1)
            if (w * h) > 260 and not_in_list_predict(box, process_con):
                crop_image = blank_image[y:y + h, x:x + w]
                image = cv2.resize(crop_image, dsize=shape)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = np.array(image).reshape(26, 45, 1)
                image = np.expand_dims(image, axis=0)
                predict = my_model.predict(image)
                name = class_name[np.argmax(predict)]
                if np.max(predict) >= 0.5 and name != 'none':
                    process_con.append((np.max(predict), name, x, y, w, h))

    print("end:", time() - start)
    new_con = np.array(process_con)

    for idx_c, box in enumerate(process_con):
        for idx_o, other in enumerate(process_con):
            predict1, text1, x1, y1, w1, h1 = box
            predict2, text2, x2, y2, w2, h2 = other
            if x1 != x2 and y1 != y2 and euclid(box, other) <= padding_size and box[1] != 'none' and other[1] != 'none':
                if predict1 > predict2:
                    new_con[idx_o][1] = 'none'
                else:
                    new_con[idx_c][1] = 'none'

    last_res = []
    for box in new_con:
        _, text, x, y, w, h = box
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        org = (x + w, y + h)
        if text != 'none':
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.putText(img, str(text), org, font,
            #             fontScale, color, thickness, cv2.LINE_AA)
            last_res.append({'text': text, 'x': x + w / 2, 'y': y + h / 2})
    # cv2.imshow('', img)
    # cv2.waitKey()
    return last_res


# my_model = load_model("modal.h5")
# using('cache/predict.jpg', my_model)
