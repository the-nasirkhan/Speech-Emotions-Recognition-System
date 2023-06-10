#!/usr/bin/env python
# coding: utf-8

# **Import Libraries**

# In[1]:


import pyaudio
import wave
import librosa
import librosa.display
import numpy as np
#import tensorflow as tf
import pickle


# **Features Extractions**

# In[2]:


# define extract_feature function
def extract_feature(file_path, mfcc=True, chroma=True, mel=True):
    with wave.open(file_path, 'rb') as wave_file:
        signal, sample_rate = librosa.load(file_path, sr=wave_file.getframerate())

    if chroma:
        stft = np.abs(librosa.stft(signal))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40).T, axis=0)
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=signal, sr=sample_rate).T, axis=0)

    return np.hstack([mfccs, chroma, mel])


# **Load Saved Model**

# In[3]:


# load the saved model
Emotion_Voice_Detection_Model = pickle.load(open("SERS_Model.pkl", 'rb'))

# record audio using microphone
CHUNK = 1024 
FORMAT = pyaudio.paInt16
CHANNELS = 1 
RATE = 44100 
RECORD_SECONDS = 4
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) 

print("* RECORDING")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data) 

print("* DONE RECORDING!")

stream.stop_stream()
stream.close()
p.terminate()

# save recorded audio to file
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


# **Apply Feature Extraction Funtion**

# In[4]:


# apply extract_feature function to the recorded file
file_path = 'output.wav'
try:
    new_feature = extract_feature(file_path)
    # predict the emotion label using the loaded model
    predicted_label = Emotion_Voice_Detection_Model.predict(np.array([new_feature]))
    print("Predicted Emotion:", predicted_label[0])
except Exception as e:
    print("Error:", str(e))


# In[ ]:




