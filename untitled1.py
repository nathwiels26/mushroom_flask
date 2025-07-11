import numpy as np
from tensorflow import keras

model = keras.models.load_model('mushroom_model.h5')
# Ganti ini dengan data yang benar2 berbeda!
x1 = np.array([[0,1,3,0,4,1,0,1,2,0,3,1]], dtype=np.float32)
x2 = np.array([[1,0,2,1,7,2,2,0,5,1,2,3]], dtype=np.float32)
print(model.predict(x1))
print(model.predict(x2))
