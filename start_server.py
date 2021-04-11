import base64
import os
import shutil
from time import time

from fastapi import FastAPI, UploadFile, File, Form
from tensorflow.keras.models import load_model
from pydantic import BaseModel

from using_modal import using

# my_model = load_model("modal.h5")
# my_model = load_model("modal_v2.h5")
# my_model = load_model("modal_v3.h5")
my_model = load_model("modal_v4.h5")


def readb64(base64_string, path='cache/predict.jpg'):
    with open(path, 'wb') as f_output:
        f_output.write(base64.b64decode(base64_string))
    return path


app = FastAPI()


class Body(BaseModel):
    image: str
    x: float
    y: float


@app.post("/for_predict")
async def root(image: Body):
    path = readb64(image.image)
    res = using(path, my_model, (image.x, image.y))
    return res
 

@app.post("/fo4_predict")
async def fo4_predict(image: UploadFile = File(...), x: float = Form(...), y: float = Form(...)):
    res = using(get_image_url(image), my_model, (x, y))
    return res


def get_image_url(image_file):
    image_folder_path = os.getcwd() + '/cache/'
    if not os.path.exists(image_folder_path):
        os.mkdir(image_folder_path)
    image_path = str(time()) + ".png"
    with open("./cache/" + image_path, "wb") as buffer:
        shutil.copyfileobj(image_file.file, buffer)
    return "./cache/" + image_path
