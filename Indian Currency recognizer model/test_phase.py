import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np


# import essentials to build cov net 

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json


# ResNet-50 model
# this is the json file that i saved in training phase
json_file = open('/kaggle/input/transfer-learning/resnet_50_model.json', 'r')

#reading model 
loaded_model_json = json_file.read()
json_file.close()

#loading model
loaded_model = model_from_json(loaded_model_json)

# loading weights into new model 
loaded_model.load_weights("../input/indian-currency-note-resnet-weights/currency_detector_2.4GB_earlyStopping_model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# These are the  class labels from the training data (Each number stands for the currency denomination)
class_labels = [
    '10','100','20','200','2000','50','500','Background'
]


#Dependecies 
#install ggts and pyttsx3 before execution

#gTTS (Google Text-to-Speech), a Python library and CLI tool to interface with Google Translate's text-to-speech API.
#Write spoken mp3 data to a file, a file-like object (bytestring) for further audio manipulation, or stdout. 

# pip install gTTS

#pyttsx3 is a text-to-speech conversion library in Python.
#Unlike alternative libraries, it works offline, and is compatible with both Python 2 and 3.

# pip install pyttsx3

# Convert the image to a numpy array
from gtts import gTTS 
from tensorflow.python.keras.preprocessing import image
import os 
import pyttsx3
    
    
def prediction(file_name):
    img = image.load_img(file_name, target_size=(256,256))

    image_to_test = image.img_to_array(img)

    #since Keras expects a list of images, not a single image,
    # Add a fourth dimension to the image 
    
    list_of_images = np.expand_dims(image_to_test, axis=0)

    # Make a prediction using the model
    results = loaded_model.predict(list_of_images)

    # Since we are only testing one image, we only need to check the first result
    single_result = results[0]

    # We will get a likelihood score for all  possible classes.
    # Find out which class had the highest score.
    # the class with highest likelihood is predicted as the result.
    
    most_likely_class_index = int(np.argmax(single_result))
    class_likelihood = single_result[most_likely_class_index]

    # Get the name of the most likely class
    class_label = class_labels[most_likely_class_index]

    # Print the result
    print(file_name)
    print("This is image is a {} - Likelihood: {: .2f}".format(class_label, class_likelihood))
    
    # convert the actual prediction result text into audio file.
    tts(class_label,class_likelihood)


# Load an image file to test, resizing it to 256x256 pixels (as required by this model)
# to save time in training I resize images to 256x256 

import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing import image

# example of test image 
img = image.load_img("../input/test-dataset/test/2000__359.jpg", target_size=(256,256))
plt.imshow(img)


# import pydub (Manipulate audio with an simple and easy high level interface)
from pydub import AudioSegment
import IPython

def tts(class_label,class_likelihood):
    language='en'
    
    # if no currency detected or uploaded image is  bagkground
    if(class_label=="Background"):
        
        mytext=' sorry but i am detecting only  the'+class_label+', please hold the note under the camera.'
    else:
        mytext="This is  {} Rs note, and I am  {: .2f} % sure of it".format(class_label, class_likelihood*100)
        
    # gTTS() converts text into the audio supports multiple languages.    
    myobj = gTTS(text=mytext, lang=language, slow=False)
    
    #store audio result 
    file='result.mp3'
    myobj.save(file)


# predict the entire test currency images 

import glob
# Find all *.jpg files in the directory
file_name_list = glob.glob('../input/test-dataset/test/*.jpg')
print(len(file_name_list))
for file_name in file_name_list:
    # print the file name 
    print(file_name)
    
    #predict the currency 
    prediction(file_name)


#predict the single image file
file_to_predict="../input/test-dataset/test/20__65.jpg"

# display currency image 
img = image.load_img(file_to_predict, target_size=(256,256))
plt.imshow(img)

#predict the currecy 

prediction(file_to_predict)

# save audio result into .mp3 file 

path='./result.mp3' 
    
IPython.display.Audio(path)