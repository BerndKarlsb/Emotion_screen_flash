# Emotion_screen_flash
This script will activate your webcam, read your facial expression and make the screen flash if you look angry for too long while working ;)

# Example Gif

# What it really does. > 2 resulting scripts (Screen_Flash_When_Angry.py & Doggy_Pic_When_Angry)
Screen_Flash_When_Angry.py activates the webcam and calls a pretrained model (Emotion_rec_model.h5) to recognice the facial expression of the user and lets your screen flash for a moment when your face looks tense.
Doggy_Pic_When_Angry.py does the same but it makes a random happy doggo pic pop up.
The camera frame rate is reduced to 1 image every approx 3 seconds to keep it efficient and once 2 consecutive images show an angry face the screen flashes instantly.

# Content
I didnt include the dataset of images for facial expressions as i wanted to keep size down to a minimum.

Screen_Flash_When_Angry.py
Use this script to activate the camera and the face recognition. ( screen flashes )

Doggy_Pic_When_Angry.py
Use this script to activate the camera and the face recognition. ( doggo pic pops up ) 

Emotion_rec_model.h5
Is the trained model with all weights and architecture saved

Classification_CNN.py 
Was used to train the model. accuracy is around 65% but as all simple emotion_rec models it depends on light/face. I dont like the model 

haarcascade_frontalface_default.xml
Is the cascade used for face recognition in openCV model

requirements.txt
Defines all packages needed
