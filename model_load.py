# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 17:07:26 2020

@author: 2ativ
"""
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', action='store', dest='model_type', 
                    help='Type VGG for VGG/n Mobile for MobileNet/n CNN for Custom Made CNN Model')
parser.add_argument('-n', action='store', dest='output_file',
                    default='output')
args = parser.parse_args()
gray_op = False
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

if args.model_type == "CNN":
    json_model = "trained_model/model.json"
    model = "trained_model/model.h5"
    gray_op = True

elif args.model_type == "VGG":
    json_model = "trained_model/vgg_model.json"
    model = "trained_model/vgg_model.h5"

elif args.model_type == "Mobile":
    json_model = "trained_model/mobilenet_model.json"
    model = "trained_model/mobilenet_model.h5"
    
    
# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

json_file = open(json_model, 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model)
print("Loaded model from disk")

args.output_file += '.avi'

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.output_file,fourcc, 20.0, (640,480))
# start the webcam feed
cap = cv2.VideoCapture(0)
print("Camera On")
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if gray_op:
            roi = gray[y:y + h, x:x + w]
        else:
            roi = frame[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi, (48, 48)), -1), 0)
        prediction = loaded_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)
        
    cv2.imshow('Press Q to exit!', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("Turning Off")
        break


cap.release()
out.release()
cv2.destroyAllWindows()