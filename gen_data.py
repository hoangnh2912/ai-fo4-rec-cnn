# import os
# from time import time
#
# import cv2
# import numpy as np
#
# shape = (26, 45)
# kernel = np.ones((2, 2), np.uint8)
#
#
# def center_box(box):
#     _, _, x, y, w, h, _ = box
#     return x + w / 2, y + h / 2
#
#
# def euclid(box1, box2):
#     x1, y1 = center_box(box1)
#     x2, y2 = center_box(box2)
#     return np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))
#
#
# def not_in_list_predict(box, list_predict):
#     _ = ""
#     x1, y1, w1, h1 = box
#     current = _, _, x1, y1, w1, h1, _
#     cx1, cy1 = center_box(current)
#     for other_pre in list_predict:
#         _, _, _, _, w2, h2, _ = other_pre
#         cx2, cy2 = center_box(other_pre)
#         if cx1 == cx2 and w2 * h2 >= w1 * h1:
#             return False
#     return True
#
#
# def gen_data_pre(path):
#     img = cv2.imread(path, 0)
#     img = cv2.resize(img, (512, 248))
#     ret, thresh_image = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
#     blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
#     edges = cv2.Canny(thresh_image, 30, 30)
#     contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#     process_con = []
#
#     for idx, con in enumerate(contours):
#         box = cv2.boundingRect(con)
#         x, y, w, h = box
#         if 10 < w * h < 2000:
#             cv2.drawContours(blank_image, contours, idx, (0, 255, 0), 1)
#             if (w * h) > 200 and not_in_list_predict(box, process_con):
#                 crop_image = blank_image[y:y + h, x:x + w]
#                 crop_image = cv2.dilate(crop_image, kernel, iterations=1)
#                 crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
#                 _, edges_crop = cv2.threshold(crop_image, 100, 255, cv2.THRESH_BINARY)
#                 cs, _ = cv2.findContours(edges_crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#                 blank_predict = np.zeros((h, w, 3), np.uint8)
#                 for i, _ in enumerate(cs):
#                     if cv2.contourArea(cs[i]) > 10:
#                         cv2.drawContours(blank_predict, cs, i, (0, 255, 0), 1)
#
#                 cv2.imwrite('cache/data_pre/' + str(time()) + ".png", blank_predict)
#
#
# for file in os.listdir('data_pre'):
#     gen_data_pre('data_pre/' + file)
#     print(file)
