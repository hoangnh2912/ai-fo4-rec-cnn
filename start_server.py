import base64
import os
import shutil
from time import time
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import cv2
from using_modal import using

# my_model = load_model("modal.h5")
# my_model = load_model("modal_v2.h5")
# my_model = load_model("modal_v3.h5")
my_model = load_model("modal_v4.h5")
print("Loaded Model")


def readb64(base64_string, path='cache/predict.jpg'):
    with open(path, 'wb') as f_output:
        f_output.write(base64.b64decode(base64_string))
    return path


app = FastAPI()


class Body(BaseModel):
    image: str
    x: float
    y: float


# @app.post("/for_predict")
# async def root(image: Body):
#     path = readb64(image.image)
#     res = using(path, my_model, (image.x, image.y))
#     return res

@app.post("/fo4_predict")
async def fo4_predict(image: UploadFile = File(...), x: float = Form(0), y: float = Form(0)):
    try:
        img = await get_mat_image(image)
        res = using(my_model, (x, y), img_cv2=img)
        return res
    except Exception as err:
        print(err)
        return {
            "error": True
        }


async def get_mat_image(image_file):
    contents = await image_file.read()
    np_arr = np.fromstring(contents, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)


def get_image_url(image_file):
    image_folder_path = os.getcwd() + '/cache/'
    if not os.path.exists(image_folder_path):
        os.mkdir(image_folder_path)
    image_path = str(time()) + ".png"
    with open("./cache/" + image_path, "wb") as buffer:
        shutil.copyfileobj(image_file.file, buffer)
    return "./cache/" + image_path
