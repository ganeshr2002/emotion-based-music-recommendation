import numpy as np
import cv2
from PIL import Image
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import *
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pandastable import Table, TableModel
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from threading import Thread
# from Spotipy import *  
import time
import pandas as pd
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor=0.6

# emotion_model = Sequential()
# emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
# emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.25))
# emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.25))
# emotion_model.add(Flatten())
# emotion_model.add(Dense(1024, activation='relu'))
# emotion_model.add(Dropout(0.5))
# emotion_model.add(Dense(7, activation='softmax'))
# emotion_model.load_weights('model.h5')

ResNet50V2 = tf.keras.applications.ResNet50V2(input_shape=(224, 224, 3),
                                               include_top= False,
                                               weights='imagenet'
                                               )
ResNet50V2.trainable = True

for layer in ResNet50V2.layers[:-50]:
    layer.trainable = False

def Create_ResNet50V2_Model():

    model = Sequential([
                      ResNet50V2,
                      Dropout(.25),
                      BatchNormalization(),
                      Flatten(),
                      Dense(64, activation='relu'),
                      BatchNormalization(),
                      Dropout(.5),
                      Dense(7,activation='softmax')
                    ])
    return model

emotion_model = Create_ResNet50V2_Model()
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
music_dist={0:"songs/angry.csv",1:"songs/disgusted.csv ",2:"songs/fearful.csv",3:"songs/happy.csv",4:"songs/neutral.csv",5:"songs/sad.csv",6:"songs/surprised.csv"}
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1 
show_text=[0]
global prev_song_features
prev_song_features = None


''' Class for calculating FPS while streaming. Used this to check performance of using another thread for video streaming '''
class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()


''' Class for using another thread for video streaming to boost performance '''
class WebcamVideoStream:
    	
		def __init__(self, src=0):
			self.stream = cv2.VideoCapture(src,cv2.CAP_DSHOW)
			(self.grabbed, self.frame) = self.stream.read()
			self.stopped = False

		def start(self):
				# start the thread to read frames from the video stream
			Thread(target=self.update, args=()).start()
			return self
			
		def update(self):
			# keep looping infinitely until the thread is stopped
			while True:
				# if the thread indicator variable is set, stop the thread
				if self.stopped:
					return
				# otherwise, read the next frame from the stream
				(self.grabbed, self.frame) = self.stream.read()

		def read(self):
			# return the frame most recently read
			return self.frame
		def stop(self):
			# indicate that the thread should be stopped
			self.stopped = True

''' Class for reading video stream, generating prediction and recommendations '''
class VideoCamera(object):
	
	def get_frame(self):
		global cap1
		global df1
		cap1 = WebcamVideoStream(src=0).start()
		image = cap1.read()
		image=cv2.resize(image,(600,500))
		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		face_rects=face_cascade.detectMultiScale(gray,1.3,5)

		file_path = "static/SoundTracks/set1_updated_tracklist.csv"

		# Load CSV into DataFrame
		df1 = pd.read_csv(file_path)

		# Define the mapping dictionary
		mapping = {
			"Angry": ["Anger"],
			"Disgusted": ["Disgust"],
			"Fearful": ["Fear"],
			"Happy": ["High Val.", "High Ener."],
			"Neutral": ["Low Val.", "Low Ener."],
			"Sad": ["Sad", "Low Val.", "Low Ener.", "Low Tens."],
			"Surprised": ["Surprise"]
		}
		emotion = emotion_dict.get(show_text[0], None)
		
		# Map emotion to genres using mapping dictionary
		available_emotions = mapping.get(emotion, [])
		
		# Filter DataFrame based on genres
		df1 = df1[df1['Emotion'].isin(available_emotions)]

		df1 = df1[['Track','Album name']]
		df1 = df1.head(15)
		for (x,y,w,h) in face_rects:
			cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
			face = gray[y:y + h, x:x + w]
			face = cv2.resize(face, (224, 224))
			face = face.astype('float32') / 255.0
			# Convert the resized face image to RGB (assuming ResNet50V2 expects RGB images)
	
			face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
			cropped_img = np.expand_dims(face, axis=0)
			prediction = emotion_model.predict(cropped_img)

			maxindex = int(np.argmax(prediction))
			show_text[0] = maxindex 
			#print("===========================================",music_dist[show_text[0]],"===========================================")
			#print(df1)
			cv2.putText(image, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
			df1 = music_rec(prev_song_features)
			
		global last_frame1
		last_frame1 = image.copy()
		pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
		img = Image.fromarray(last_frame1)
		img = np.array(img)
		ret, jpeg = cv2.imencode('.jpg', img)
		return jpeg.tobytes(), df1

def load_audio_for_emotion_and_genre(emotion, df):
    filtered_df = df[(df['Emotion'] == emotion)]
    return filtered_df

def music_rec():
	# print('---------------- Value ------------', music_dist[show_text[0]])
	# df = pd.read_csv(music_dist[show_text[0]])
	# df = df[['Name','Album','Artist']]
	# df = df.head(15)
	# return df

	file_path = "static/SoundTracks/set1_updated_tracklist.csv"

	# Load CSV into DataFrame
	df = pd.read_csv(file_path)
	db_emotions = ['Happy', 'Sad', 'Tender', 'Fear', 'Anger', 'Surprise', 'High Val.', 'Low Val.', 'High Ener.', 'Low Ener.', 'High Tens.', 'Low Tens.']

	# Define the mapping dictionary
	mapping = {
		"Angry": ["Anger"],
		"Disgusted": ["Disgust"],
		"Fearful": ["Fear"],
		"Happy": ["High Val.", "High Ener."],
		"Neutral": ["Low Val.", "Low Ener."],
		"Sad": ["Sad", "Low Val.", "Low Ener.", "Low Tens."],
		"Surprised": ["Surprise"]
	}
	predicted_emotion = emotion_dict.get(show_text[0], None)
    
    # Map emotion to genres using mapping dictionary
	available_emotions = mapping.get(predicted_emotion, [])
    
    # Filter DataFrame based on genres
	filtered_df = df[df['Emotion'].isin(available_emotions)]
	filtered_df = filtered_df.sort_values(by='genre')
	musicDir = "static/SoundTracks/Set1/Set1"


	    # Calculate similarity if prev_song_features is not Non
	if prev_song_features is not None:
		similarities = []
		print('hi')
		
		for index, row in filtered_df.iterrows():
            # Extract features for the current song
			audioPath = musicDir + "/" + row['Nro'] + ".mp3"
			current_song_features = extract_features(audioPath)
            # Calculate cosine similarity between prev_song_features and current_song_features
			similarity = cosine_similarity(prev_song_features, current_song_features)
			similarities.append(similarity[0][0])
        
        # Add a new column 'Similarity' to the DataFrame
		filtered_df['Similarity'] = similarities
        # Sort DataFrame based on similarity
		filtered_df = filtered_df.sort_values(by='Similarity', ascending=False)


	return filtered_df

def extract_features(file_path):
    x, sr = librosa.load(file_path)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20)

    # Take the mean along each feature dimension
    mfcc_mean = np.mean(mfcc, axis=1)

    features = mfcc_mean.reshape(1, 1, 20)  # Reshape to match the model input shape
    return features