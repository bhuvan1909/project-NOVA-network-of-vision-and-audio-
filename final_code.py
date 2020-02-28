#!/usr/bin/env python
# coding: utf-8
#by M.BHUVAN SUNDHAR REDDY
# # Object Detection 
# Welcome to the object detection inference walkthrough!

# # Imports

# In[ ]:
import numpy as np
import wikipedia
import os
import six.moves.urllib as urllib
import sys
import tarfile
import pandas as pd
import tensorflow as tf
import cv2
import pyttsx3
import datetime
import speech_recognition as sr 
import smtplib,ssl

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops 
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate',180)


if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')


# ## Env setup

# In[ ]:


# This is needed to display the images.
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Object detection imports
# Here are the imports from the object detection module.

# In[ ]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[ ]:


# What model to download.
MODEL_NAME = os.path.join('cocotrained_models','ssd_mobilenet_v1_coco_2017_11_17')#keep the location of downloaded model file.
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/cocotrained_models'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')


# ## Download Model

# In[ ]:



file_name = os.path.join(MODEL_NAME,'frozen_inference_graph.pb')
  
file_name= os.getcwd()


# ## Load a (frozen) Tensorflow model into memory.

# In[ ]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[ ]:

NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



# ## Helper code

# In[ ]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("Good Morning!")

    elif hour>=12 and hour<18:
        speak("Good Afternoon!")   

    else:
        speak("Good Evening!")  

    speak("I am AI. Please tell me how may I help you")     
def sendEmail(to, content):
    port = 465
    sender='xxxxxxxxxxxxx@gmail.com'  #put email from which account to send from
    password='xxxxxxx'  #password for email
      
    context = ssl.create_default_context()
    print("Starting to send")
    speak("starting to send!")
    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender, password)
        server.sendmail(sender, to,content)

    print("sent email!")
    speak("sent mail!")
    engine.runAndWait()
    server.close()
def emailcommand():
    #It takes microphone input from the user and returns string output

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")    
        command = r.recognize_google(audio, language='en-in')
        print(f"User said: {command}\n")
        
    except Exception as e:
        # print(e)    
        print("Say that again please...")  
        command = emailcommand()
    return command
def takeCommand():
    #It takes microphone input from the user and returns string output

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")    
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")
        if 'email' not in query and'detect'not in query and 'the time' not in query and 'Wikipedia'not in query and 'shutdown'not in query and 'logoff'not in query and 'go to sleep' not in query:
            query=takeCommand()
    except Exception as e:
        # print(e)    
        print("Say that again please...")  
        query = takeCommand()
    


    
 
# # Detection

# In[ ]: 
    
       
    while 'detect' in query:
        cam = cv2.VideoCapture(0)
        while True:
           ret, frame = cam.read()
            
           if not ret:
                print("unable to use camera")
                engine.say("unable to use camera")
                break
           cv2.namedWindow("test")
           cv2.imshow("test", frame)
           k = cv2.waitKey(1)
          
           if k%256 == 27:
                # ESC pressed
            print("Escape hit, closing...")
            engine.say("closing all,Good bye sir!")
            cv2.destroyAllWindows()
            cam.release()
            exit()
                
           elif k%256 == 32 :
                # SPACE pressed
            img_name = "F:/tensorflow/models/research/object_detection/test_images/image.jpg"#keep location where to store picture
           
            cv2.imwrite(img_name, frame)
            cv2.destroyAllWindows()
            
            print("image saved")
            engine.say("image saved.")
            engine.runAndWait()
            break
             
        cam.release()  
        
        
        # For the sake of simplicity we will use only 2 images:
        # image1.jpg
        # image2.jpg
        # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
        PATH_TO_TEST_IMAGES_DIR = 'test_images'
        image_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image.jpg')  
        
        # Size, in inches, of the output images.
        
        
        
        # In[ ]:
        
        
        def run_inference_for_single_image(image, graph):
          with graph.as_default():
            with tf.compat.v1.Session() as sess:
              # Get handles to input and output tensors
              ops = tf.compat.v1.get_default_graph().get_operations()
              all_tensor_names = {output.name for op in ops for output in op.outputs}
              tensor_dict = {}
              for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
              ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                  tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                      tensor_name)
              if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.60), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
              image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')
        
              # Run inference
              output_dict = sess.run(tensor_dict,
                                     feed_dict={image_tensor: image})
        
              # all outputs are float32 numpy arrays, so convert types as appropriate
              output_dict['num_detections'] = int(output_dict['num_detections'][0])
              output_dict['detection_classes'] = output_dict[
                  'detection_classes'][0].astype(np.uint8)
              output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
              output_dict['detection_scores'] = output_dict['detection_scores'][0]
              if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
              
          return output_dict
        
        
         # In[ ]:
        
        dataframe=open("savedobjects.csv",'w')
        savedclass=[]
        
        image = Image.open(image_path)
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
        output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
          
          # Visualization of the results of a detection
         
        nprehit = output_dict['detection_scores'].shape[0]
        engine.say("detected objects are ")
        engine.runAndWait()
        print("detected objects are:")
        for j in range(nprehit):
         score = (output_dict['detection_scores'][j])
         if score>0.60:
            fname = "image"+str(1)
            classid = int(output_dict['detection_classes'][j])
            classname = category_index[classid]["name"]
            print(classname)
            
            
            savedclass.append(classname)
        
        engine.say(savedclass)  
        engine.runAndWait()  
        dataframe1=pd.DataFrame(savedclass)
        dataframe1.to_csv("savedobjectsnet.csv",index =False,header=False)
        print("csv file saved")
        print("exiting..")
        query=takeCommand()
        
             # Detected_objects.to_csv("savedobjectsnet.csv",mode='a',index =True,sep=',', header=True)
           #print(output_dict['detection_boxes'].shape)
           #print(output_dict['num_detections'])
    if 'the time' in query:
                strTime = datetime.datetime.now().strftime("%H:%M:%S")    
                speak(f"Sir, the time is {strTime}")
                query = takeCommand()
                
    if 'Wikipedia' in query:
                speak('Searching Wikipedia...')
                query = query.replace("Wikipedia", "")
                results = wikipedia.summary(query, sentences=2)
                speak("According to Wikipedia")
                print(results)
                speak(results)   
                query = takeCommand()
    if 'email' in query:
            try:
                speak("What should I say?")
                content = emailcommand()
                to = ['xxxxxxx@gmail.com'] #replace with ur email 
                sendEmail(to, content)
                takeCommand()
            except Exception as e:
                print(e)
                speak("Sorry I am not able to send this email") 
                engine.runAndWait()
                takeCommand()
          
    elif 'shutdown'in query or 'logoff'in query or 'go to sleep' in query:
        engine.say("good bye sir!,Have a nice day.")
        engine.runAndWait()
        cv2.destroyAllWindows()
        sys.exit()
wishMe()
takeCommand()    
       
    
    
