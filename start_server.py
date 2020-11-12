import base64

from fastapi import FastAPI
from pydantic import BaseModel
from keras.models import load_model
from using_modal_test_server import using

my_model = load_model("modal.h5")


def readb64(base64_string):
    path = 'cache/predict.jpg'
    with open(path, 'wb') as f_output:
        f_output.write(base64.b64decode(base64_string))
    return path


app = FastAPI()


class Body(BaseModel):
    image: str


@app.post("/for_predict")
async def root(image: Body):
    path = readb64(image.image)
    res = using(path, my_model)
    return res
