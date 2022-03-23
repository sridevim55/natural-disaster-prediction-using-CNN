

import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model = load_model(r"E:\mini_Project\project1\Final_ Project _ Natural_Disaster_Prediction/Natural_Disaster_Prediction.h5")
                 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (50,50)) 
        x = image.img_to_array(img)
        print(x)
        x = np.expand_dims(x,axis=0)
        print(x)
        preds = model.predict(x)
        pred=np.argmax(preds,axis=1)
        print("prediction",pred)
        index = ['Cyclone','Earthquake','Flood','Wildfire']
        text = "You are " + str(index[pred[0]])
    return text
if __name__ == '__main__':
    app.run(debug = False, threaded = False)