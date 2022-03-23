import numpy as np
#import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

x_train=train_datagen.flow_from_directory(r"E:\mini_Project\project1\Final_ Project _ Natural_Disaster_Prediction\train_set", target_size=(50,50), batch_size=32, class_mode="categorical")

x_test=train_datagen.flow_from_directory(r"E:\mini_Project\project1\Final_ Project _ Natural_Disaster_Prediction\test_set", target_size=(50,50), batch_size=32, class_mode="categorical")

print(x_train.class_indices)

model=Sequential()

model.add(Convolution2D(32,(3,3), input_shape=(50,50,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128,activation="relu"))

model.add(Dense(units=4,activation="softmax"))

model.summary()
 
model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])

# steps_per_epoch=no.of images in train data/batch_size
#19455/32=608
# validation steps=no.of images in test data/batch _size
#7030/32=220

model.fit(x_train,steps_per_epoch=23,epochs=20, validation_data=x_test, validation_steps=20)

model.save('Natural_Disaster_Prediction.h5')

from tensorflow.keras.models import load_model
#from tensorflow.keras.models import Sequential
from keras.preprocessing import image

import numpy as np
 
mymodel=load_model(r"E:\mini_Project\project1\Final_ Project _ Natural_Disaster_Prediction\Natural_Disaster_Prediction.h5")

img=image.load_img(r"E:\mini_Project\project1\Final_ Project _ Natural_Disaster_Prediction\train_set\Wildfire\1.jpg",target_size=(50,50))
img

xx1=image.img_to_array(img)

xx1

xx1.shape

xx2=np.expand_dims(xx1,axis=0)

xx2.shape

#pred=mymodel.predict_class(xx2)
pred=np.argmax(mymodel)

print(pred)

y=mymodel.predict(xx2)
pred=np.argmax(y,axis=1)
print(pred)

print(x_train.class_indices )

index=['Cyclone','Earthquake','Flood','Wildfire']
result=str(index[pred[0]])
print(result)
