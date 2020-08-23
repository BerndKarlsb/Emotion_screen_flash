###########################################################
# This script here is a modification of the Screen_Flash_When_Angry script
# This one here shows a random image of a dog when you look angry during work
###########################################################

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import time
from bs4 import BeautifulSoup
import ast
import random
import requests

###########################################################
# Getting a list of images to randomly choose from
# method: use base_url, choose random dog image from list and add it to base_url
# source : www.Random.dog
###########################################################

result = requests.get("https://random.dog/doggos")
source = result.content
soup = BeautifulSoup(source, "lxml")
body = soup.find('body')
body = body.text
doggy_image_list = ast.literal_eval(body)
doggy_image_list  = [x for x in doggy_image_list if "mp4" not in x]
doggy_image_list  = [x for x in doggy_image_list if "gif" not in x]
base_url = "https://random.dog/"

###########################################################
# Get randomn element from image list, join with base_url and open image and resize
###########################################################

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def url_to_image():
	random_doggy = random.choice(doggy_image_list)
	url = base_url + random_doggy
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = requests.get(url, stream=True).raw
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	image = image_resize(image, height = 700)
	# return the image
	return image

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
					count = count + 1
					if count == 2:
						image = url_to_image()
						(h, w) = image.shape[:2]
						cv2.namedWindow( "doggy")
						cv2.resizeWindow("doggy", h, w)
						cv2.imshow("doggy", image)
						cv2.waitKey(3000)
						cv2.destroyAllWindows()
						count = count + 1
					if count > 2:
						count = 0
				else:
					count = 0
		# # in case you want to see the screen image
		# cv2.imshow('Emotion Detector', frame)
		# if cv2.waitKey(1) & 0xFF == ord('q'):
		# 	break

cap.release()
cv2.destroyAllWindows()
