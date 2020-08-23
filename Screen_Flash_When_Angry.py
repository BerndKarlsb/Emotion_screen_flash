###########################################################
# This script here is a lot more useful than the doggy images popping up
# it just blinks shortly when your face seems tense
###########################################################

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from win32com.client import GetObject
import wmi
import time

###########################################################
# Get starting point brightness to revert to after the flash
###########################################################

objWMI = GetObject('winmgmts:\\\\.\\root\\WMI').InstancesOf('WmiMonitorBrightness')
for obj in objWMI:
	startpointbrightness = obj.CurrentBrightness
print(startpointbrightness)

###########################################################
# variables to later adjust screen brightness
###########################################################

c = wmi.WMI(namespace='wmi')
methods = c.WmiMonitorBrightnessMethods()[0]

##########################################################
# Load emotion recognition model
##########################################################

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('Emotion_rec_model.h5')

##########################################################
# define class labels
##########################################################

class_labels = ['Angry','Happy','Neutral','Sad','Surprise', 'Fear', 'Digust']

##########################################################
# Start video capture and define frame rate and time
##########################################################

frame_rate = 0.15
prev = 0
cap = cv2.VideoCapture(0)
count = 0

while True:

	# Grab a single frame of video every frame-rate

	time_elapsed = time.time() - prev
	ret, frame = cap.read()

	if time_elapsed > 1./frame_rate:
		prev = time.time()

		# now start processing

		labels = []
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces = face_classifier.detectMultiScale(gray,1.3,5)

		for (x,y,w,h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)

			# rect,face,image = face_detector(frame)

			if np.sum([roi_gray])!=0:
				roi = roi_gray.astype('float')/255.0
				roi = img_to_array(roi)
				roi = np.expand_dims(roi,axis=0)

				# make a prediction on the ROI, then lookup the class,

				preds = classifier.predict(roi)[0]
				label = class_labels[preds.argmax()]
				label_position = (x, y)

				# if prediction is Fear or Angry, read screen brightness, if no set back to earlier state (auto adjust in my case)
				# for some reason fear is really close to angry in my case... and my standard face is "surprised" so... choose whatever labels works better for you
				# all code commented out in the next lines is there in case you want to visually check what label corresponds to which facial expression.

				if label == "Angry" or label == "Fear":
					cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
					count = count + 1
					if count == 2:
						methods.WmiSetBrightness(100, 1)
						time.sleep(0.2)
						methods.WmiSetBrightness(10, 1)
						time.sleep(0.3)
						methods.WmiSetBrightness(startpointbrightness, 1)
						time.sleep(0.2)
					if count > 2:
						count = 0
				else:
					cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
					count = 0
		# in case you want to see the screen image
		cv2.imshow('Emotion Detector', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()
