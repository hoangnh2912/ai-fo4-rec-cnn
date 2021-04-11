from time import time

import cv2
import numpy as np
import os

# shape = (26, 45)
shape = (21, 39)
class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'none']
padding_size = 50
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (0, 255, 0)
thickness = 1
kernel = np.ones((2, 2), np.uint8)


def center_box(box):
    _, _, x, y, w, h, _ = box
    return x + w / 2, y + h / 2


def euclid(box1, box2):
    x1, y1 = center_box(box1)
    x2, y2 = center_box(box2)
    return np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))


def not_in_list_predict(box, list_predict):
    _ = ""
    x1, y1, w1, h1 = box
    current = _, _, x1, y1, w1, h1, _
    cx1, cy1 = center_box(current)
    for other_pre in list_predict:
        _, _, _, _, w2, h2, _ = other_pre
        cx2, cy2 = center_box(other_pre)
        if cx1 == cx2 and w2 * h2 >= w1 * h1 and cy1 == cy2:
            return False
    return True


async def using(path, my_model, add=(0, 0)):
    img = cv2.imread(path, 0)
    h_ori, w_ori = img.shape
    img = cv2.resize(img, (512, 248), interpolation=cv2.INTER_LANCZOS4)
    ret, thresh_image = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    edges = cv2.Canny(thresh_image, 1, 1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    process_con = []

    start = time()
    for idx, con in enumerate(contours):
        box = cv2.boundingRect(con)
        x, y, w, h = box
        if 10 < w * h < 2000:
            cv2.drawContours(blank_image, contours, idx, (0, 255, 0), 1)
            # if (w * h) > 150:
            if (w * h) > 150 and not_in_list_predict(box, process_con):
                crop_image = blank_image[y:y + h, x:x + w]
                crop_image = cv2.dilate(crop_image, kernel, iterations=1)
                crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
                _, edges_crop = cv2.threshold(crop_image, 5, 255, cv2.THRESH_BINARY)
                cs, _ = cv2.findContours(edges_crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                blank_predict = np.zeros((h, w, 3), np.uint8)
                for i, _ in enumerate(cs):
                    if cv2.contourArea(cs[i]) > 15:
                        cv2.drawContours(blank_predict, cs, i, (0, 255, 0), 1)
                blank_predict = cv2.cvtColor(blank_predict, cv2.COLOR_BGR2GRAY)

                image = cv2.resize(blank_predict, dsize=shape, interpolation=cv2.INTER_NEAREST)
                image_dup = cv2.resize(blank_predict, dsize=shape, interpolation=cv2.INTER_LANCZOS4)
                # image = np.array(image).reshape(26, 45, 1)
                image = np.array(image).reshape(21, 39, 1)
                image_dup = np.array(image_dup).reshape(21, 39, 1)
                # image = np.expand_dims(image, axis=0)
                predict = my_model.predict(np.array([image, image_dup]))
                arg_max_nearest = np.argmax(predict[0])
                arg_max_lanczos4 = np.argmax(predict[1])
                arg_predict = arg_max_nearest
                if arg_max_nearest != arg_max_lanczos4:
                    if arg_max_nearest == 10:
                        arg_predict = arg_max_lanczos4
                    if arg_max_lanczos4 == 10:
                        arg_predict = arg_max_nearest
                name = class_name[arg_predict]
                # print(name)
                # cv2.imshow(name, blank_predict)
                # cv2.waitKey()
                # print()
                if np.max(predict) >= 0.4 and name != 'none' and w <= h:
                    process_con.append((np.max(predict), name, x, y, w, h, idx))
    new_con = np.array(process_con)
    for idx_c, box in enumerate(process_con):
        for idx_o, other in enumerate(process_con):
            predict_box, text_box, x1, y1, w1, h1, idx_box = box
            predict_other, text_other, x2, y2, w2, h2, idx_other = other
            if x1 != x2 and y1 != y2 and euclid(box, other) <= padding_size and box[1] != 'none' and other[1] != 'none':
                # if predict_box > predict_other:
                if cv2.contourArea(contours[idx_box]) > cv2.contourArea(contours[idx_other]):
                    new_con[idx_o][1] = 'none'
                else:
                    new_con[idx_c][1] = 'none'

    last_res = []
    for box in new_con:
        _, text, x, y, w, h, i = box
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        org = (x + w, y + h)
        if text != 'none':
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(img, str(text), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
            cv2.drawContours(blank_image, contours, int(i), (0, 255, 255), 1)
            x_add, y_add = add
            w_re, h_re = (512, 248)
            res_x = x_add + (x + w / 2) * (w_ori / w_re)
            res_y = y_add + (y + h / 2) * (h_ori / h_re)
            last_res.append({'text': text, 'x': res_x, 'y': res_y})

    print("end:", time() - start, "text:", last_res)
    os.system("rm -rf " + path)
    # cv2.imshow("blank_image", blank_image)
    # cv2.imshow('', img)
    # cv2.waitKey()

    if len(last_res) == 0:
        return {
            "error": "no result"
        }
    return last_res
