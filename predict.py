from sys import argv
from tensorflow.keras.models import load_model
from CONST import IMG_SHAPE, MAP
from tensorflow.keras.preprocessing import image
import numpy as np

import os

model = None

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    if len(argv) != 3 :
        print("Please enter all required arguments")
    
    else:
        model = None
        img = None

        try:
            model = load_model(argv[1])
        except:
            raise Exception("Unable to load model")
        
        try:
            img = image.load_img(argv[2], target_size = IMG_SHAPE)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
        except:
            raise Exception("Unable to load image")

        try:
            prediction = model.predict(img)[0]
            ndx = [i for i, prd in enumerate(prediction) if prd>0.5]
            print(MAP[ndx[0]])
        except:
            raise Exception("Unable to predict")
