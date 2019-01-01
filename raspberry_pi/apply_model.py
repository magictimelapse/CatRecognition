from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import  preprocess_input
import numpy as np
img_width, img_height = 299, 299
print('loading module... Please wait...')
model = load_model('inceptionv3_1.h5')
print (model.summary())
def predict(image_file):
    img = image.load_img(image_file, target_size=(img_width,img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds
