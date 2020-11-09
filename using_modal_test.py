import cv2
import numpy as np
from keras.models import load_model

class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'none']
my_model = load_model("modal.h5")
shape = (26, 45)
path = 'data_pre/image_56.jpg'
img = cv2.imread(path, 0)
img_origin = cv2.imread(path)
#
# img = cv2.resize(img, (250, 122))
# img_origin = cv2.resize(img_origin, (250, 122))

img = cv2.resize(img, (512, 248))
img_origin = cv2.resize(img_origin, (512, 248))

blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
edges = cv2.Canny(img, 100, 100)
contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

process_con = []

for idx, con in enumerate(contours):
    x, y, w, h = cv2.boundingRect(con)
    if 40 < w * h < 2000:
        cv2.drawContours(blank_image, contours, idx, (0, 255, 0), 1)
        if (w * h) > 260:
            crop_image = blank_image[y:y + h, x:x + w]
            image = cv2.resize(crop_image, dsize=shape)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.array(image).reshape(26, 45, 1)
            image = np.expand_dims(image, axis=0)
            predict = my_model.predict(image)
            name = class_name[np.argmax(predict)]
            if np.max(predict) >= 0.7 and name != 'none':
                process_con.append((np.max(predict), name, x, y, w, h))


def center_box(box):
    _, _, x, y, w, h = box
    return x + w / 2, y + h / 2


def euclid(box1, box2):
    x1, y1 = center_box(box1)
    x2, y2 = center_box(box2)
    return np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))


new_con = np.array(process_con)

for idx_c, con in enumerate(process_con):
    for idx_o, other in enumerate(process_con):
        p1, t1, x1, y1, w1, h1 = con
        p2, t2, x2, y2, w2, h2 = other
        if x1 != x2 and y1 != y2 and euclid(con, other) <= 50 and con[1] != 'none' and other[1] != 'none':
            # p1, _, _, _, w1, h1 = con
            # p2, _, _, _, w2, h2 = other
            print(p1, t1, " - ", p2, t2)

            # if w1 * h1 > w2 * h2:
            if p1 > p2:
                new_con[idx_o][1] = 'none'
            else:
                new_con[idx_c][1] = 'none'
for con in new_con:
    _, text, x, y, w, h = con
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (x + w, y + h)
    fontScale = 0.5
    color = (0, 255, 0)
    thickness = 1
    if text != 'none':
        cv2.putText(img_origin, str(text), org, font,
                    fontScale, color, thickness, cv2.LINE_AA)
        cv2.rectangle(img_origin, (x, y), (x + w, y + h), (0, 255, 255), 1)

cv2.imshow("Picture", img_origin)
cv2.waitKey(0)
