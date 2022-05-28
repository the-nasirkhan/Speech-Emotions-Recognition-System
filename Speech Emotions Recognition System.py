#!/usr/bin/env python
# coding: utf-8

# # SPEECH EMOTION RECOGNITION SYSTEM

# **PROJECT OVERVIEW**
# 
# Through all the available senses humans can actually sense the emotional state of their communication partner. The emotional detection is natural for humans but it is very difficult task for computers; although they can easily understand content based information, accessing the depth behind content is difficult and that’s what speech emotion recognition (SER) sets out to do. It is a system through which various audio speech files are classified into different emotions such as happy, sad, anger and neutral by computer. SER can be used in areas such as the medical field or customer call centers.
# 
# **DATASET**
# 
# The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) Dataset from Kaggle contains 1440 audio files from 24 Actors vocalizing two lexically-matched statements. Emotions include angry, happy, sad, fearful, calm, neutral, disgust, and surprised.[Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

# ### STEP 1 - IMPORT DEPENDENCIES LIBRARIES

# In[1]:


#Install all the Reqiuired Libraries and Packages 
import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc , logfbank
import pickle
import librosa
from scipy import signal
import noisereduce as nr
import librosa.display
import soundfile as sf
from IPython.display import Audio
get_ipython().magic('matplotlib inline')
import soundfile
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter("ignore")


# ### STEP 2 - LOAD THE RAVDESS DATASET

# In[2]:


#Loading the required RAVDESS DataSet with length of 1439 Audio Files 
os.listdir(path='.\speech-emotion-recognition-ravdess-data')
def getListOfFiles(dirName):
    listOfFile=os.listdir(dirName)
    allFiles=list()
    for entry in listOfFile:
        fullPath=os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles=allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

dirName = './speech-emotion-recognition-ravdess-data'
listOfFiles = getListOfFiles(dirName)
len(listOfFiles)


# **Male Neutral**

# In[24]:


#Male Neutral

# LOAD IN FILE
x, sr = librosa.load('D:/Minor Project/speech-emotion-recognition-ravdess-data/Actor_01/03-01-01-01-01-01-01.wav')

# DISPLAY WAVEPLOT
plt.figure(figsize=(8, 4))
librosa.display.waveshow(x, sr=sr)
plt.title('Waveplot - Male Neutral')
plt.savefig('Waveplot_MaleNeutral.png')

# PLAY AUDIO FILE

sf.write('D:/Minor Project/speech-emotion-recognition-ravdess-data/Actor_01/03-01-01-01-01-01-01.wav', x,sr)
Audio(data=x, rate=sr)


# **Female Calm**

# In[25]:


#Female Calm

# LOAD IN FILE
x, sr = librosa.load('D:/Minor Project/speech-emotion-recognition-ravdess-data/Actor_02/03-01-02-01-01-01-02.wav')

# DISPLAY WAVEPLOT
plt.figure(figsize=(8, 4))
librosa.display.waveshow(x, sr=sr)
plt.title('Waveplot - Female Calm')
plt.savefig('Waveplot_FemaleCalm.png')

# PLAY AUDIO FILE
sf.write('D:/Minor Project/speech-emotion-recognition-ravdess-data/Actor_02/03-01-02-01-01-01-02.wav', x,sr)
Audio(data=x, rate=sr)


# **Male Happy**

# In[26]:


#Male Happy

# LOAD IN FILE
x, sr = librosa.load('D:/Minor Project/speech-emotion-recognition-ravdess-data/Actor_03/03-01-03-01-01-01-03.wav')

# DISPLAY WAVEPLOT
plt.figure(figsize=(8, 4))
librosa.display.waveshow(x, sr=sr)
plt.title('Waveplot - Male Happy')
plt.savefig('Waveplot_MaleHappy.png')

# PLAY AUDIO FILE

sf.write('D:/Minor Project/speech-emotion-recognition-ravdess-data/Actor_03/03-01-03-01-01-01-03.wav', x,sr)
Audio(data=x, rate=sr)


# **Female Sad**

# In[27]:


#Female Sad

# LOAD FILE
x, sr = librosa.load('D:/Minor Project/speech-emotion-recognition-ravdess-data/Actor_04/03-01-04-01-01-01-04.wav')

# DISPLAY WAVEPLOT
plt.figure(figsize=(8, 4))
librosa.display.waveshow(x, sr=sr)
plt.title('Waveplot - Female Sad')
plt.savefig('Waveplot_FemaleSad.png')

# PLAY AUDIO FILE
sf.write('D:/Minor Project/speech-emotion-recognition-ravdess-data/Actor_04/03-01-04-01-01-01-04.wav', x, sr)
Audio(data=x, rate=sr)


# **Male Angry**

# In[28]:


#Male Angry

# LOAD FILE
x, sr = librosa.load('D:/Minor Project/speech-emotion-recognition-ravdess-data/Actor_05/03-01-05-02-02-01-05.wav')

# DISPLAY WAVEPLOT
plt.figure(figsize=(8, 4))
librosa.display.waveshow(x, sr=sr)
plt.title('Waveplot - Male Angry')
plt.savefig('Waveplot_MaleAngry.png')

# PLAY AUDIO FILE
sf.write('D:/Minor Project/speech-emotion-recognition-ravdess-data/Actor_05/03-01-05-01-01-01-02.wav', x, sr)
Audio(data=x, rate=sr)


# **Female Fearful**

# In[29]:


#Female Fearful

# LOAD FILE
x, sr = librosa.load('D:/Minor Project/speech-emotion-recognition-ravdess-data/Actor_06/03-01-06-01-01-01-06.wav')

# DISPLAY WAVEPLOT
plt.figure(figsize=(8, 4))
librosa.display.waveshow(x, sr=sr)
plt.title('Waveplot - Female Fearful')
plt.savefig('Waveplot_FemaleFearful.png')

# PLAY AUDIO FILE
sf.write('D:/Minor Project/speech-emotion-recognition-ravdess-data/Actor_06/03-01-06-01-01-01-06.wav', x, sr)
Audio(data=x, rate=sr)


# **Male Disgust**

# In[30]:


#Male Disgust

# LOAD FILE
x, sr = librosa.load('D:/Minor Project/speech-emotion-recognition-ravdess-data/Actor_07/03-01-07-01-01-01-07.wav')

# DISPLAY WAVEPLOT
plt.figure(figsize=(8, 4))
librosa.display.waveshow(x, sr=sr)
plt.title('Waveplot - Male Disgust')
plt.savefig('Waveplot_MaleDisgust.png')

# PLAY AUDIO FILE
sf.write('D:/Minor Project/speech-emotion-recognition-ravdess-data/Actor_07/03-01-07-01-01-01-07.wav', x, sr)
Audio(data=x, rate=sr)


# **Female Surprised**

# In[31]:


#Female Surprised

# LOAD FILE
x, sr = librosa.load('D:/Minor Project/speech-emotion-recognition-ravdess-data/Actor_08/03-01-08-01-01-01-08.wav')

# DISPLAY WAVEPLOT
plt.figure(figsize=(8, 4))
librosa.display.waveshow(x, sr=sr)
plt.title('Waveplot - FemaleSurprised')
plt.savefig('Waveplot_FemaleSurprised.png')

# PLAY AUDIO FILE
sf.write('D:/Minor Project/speech-emotion-recognition-ravdess-data/Actor_08/03-01-08-01-01-01-08.wav', x, sr)
Audio(data=x, rate=sr)


# ### STEP 3 - USING SPEECH RECOGNITION API TO CONVERT AUDIO TO TEXT

# In[3]:


#Use the Speech-Recognition API to get the Raw Text from Audio Files, Though Speech Recognition
#is less strong for large chunk of files , so used Error Handling , where when it is not be able to 
#produce the text of a particular Audio File it prints the statement 'error'.Just for understanding Audio
import speech_recognition as sr
r=sr.Recognizer()
for file in range(0 , len(listOfFiles) , 1):
    with sr.AudioFile(listOfFiles[file]) as source:
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print(text)
        except:
            print('error')


# ### STEP 4 -  PLOTTING TO UNDERSTAND RAW AUDIO FILES

# In[4]:



def envelope(y , rate, threshold):
   mask=[]
   y=pd.Series(y).apply(np.abs)
   y_mean = y.rolling(window=int(rate/10) ,  min_periods=1 , center = True).mean()
   for mean in y_mean:
       if mean>threshold:
           mask.append(True)
       else:
           mask.append(False)
   return mask


# In[5]:


#Plotting the Basic Graphs for understanding of Audio Files :
for file in range(0 , len(listOfFiles) , 1):
    audio , sfreq = librosa.load(listOfFiles[file])
    time = np.arange(0 , len(audio)) / sfreq
    
    fig ,ax = plt.subplots()
    ax.plot(time , audio)
    ax.set(xlabel = 'Time (s)' , ylabel = 'Sound Amplitude')
    plt.show()
    
#PLOT THE SEPCTOGRAM
for file in range(0 , len(listOfFiles) , 1):
     sample_rate , samples = wavfile.read(listOfFiles[file])
     frequencies , times, spectrogram = signal.spectrogram(samples, sample_rate) 
     plt.pcolormesh(times, frequencies, spectrogram)
     plt.imshow(spectrogram)
     plt.ylabel('Frequency [Hz]')
     plt.xlabel('Time [sec]')
     plt.show()


# ### STEP 5 - VISUALIZATION OF AUDIO DATA

# In[6]:


#Next Step is In-Depth Visualisation of Audio Fiels and its certain features to plot for.
#They are the Plotting Functions to be called later. 
def plot_signals(signals):
    fig , axes = plt.subplots(nrows=2, ncols=5,sharex =False , sharey=True, figsize=(20,5))
    fig.suptitle('Time Series' , size=16)
    i=0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i +=1

def plot_fft(fft):
    fig , axes = plt.subplots(nrows=2, ncols=5,sharex =False , sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transform' , size=16)
    i=0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y,freq = data[0] , data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq , Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i +=1
    
def plot_fbank(fbank):
    fig , axes = plt.subplots(nrows=2, ncols=5,sharex =False , sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients' , size=16)
    i=0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],cmap='hot', interpolation = 'nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i +=1
            
def plot_mfccs(mfccs):
    fig , axes = plt.subplots(nrows=2, ncols=5,sharex =False , sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Capstrum  Coefficients' , size=16)
    i=0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                             cmap='hot', interpolation = 'nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i +=1

def calc_fft(y,rate):
    n = len(y)
    freq = np.fft.rfftfreq(n , d= 1/rate)
    Y= abs(np.fft.rfft(y)/n)
    return(Y,freq)


# In[7]:


# Here The Data Set is loaded and plots are Visualised by Calling the Plotting Functions . 
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
for file in range(0 , len(listOfFiles) , 1):
    rate, data = wav.read(listOfFiles[file])
    fft_out = fft(data)
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.plot(data, np.abs(fft_out))
    plt.show()
    
signals={}
fft={}
fbank={}
mfccs={}
# load data
for file in range(0 , len(listOfFiles) , 1):
#     rate, data = wavfile.read(listOfFiles[file])
     signal,rate =librosa.load(listOfFiles[file] , sr=44100)
     mask = envelope(signal , rate , 0.0005)
     signals[file] = signal
     fft[file] = calc_fft(signal , rate)
    
     bank = logfbank(signal[:rate] , rate , nfilt = 26, nfft = 1103).T
     fbank[file] = bank
     mel = mfcc(signal[:rate] , rate , numcep =13 , nfilt = 26 , nfft=1103).T
     mfccs[file]=mel

plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()

plot_fbank(fbank)
plt.show()

plot_mfccs(mfccs)
plt.show()


# ### STEP 6 - CLEANING & MASKING

# In[8]:


#Now Cleaning Step is Performed where:
#DOWN SAMPLING OF AUDIO FILES IS DONE  AND PUT MASK OVER IT AND DIRECT INTO CLEAN FOLDER
#MASK IS TO REMOVE UNNECESSARY EMPTY VOIVES AROUND THE MAIN AUDIO VOICE 
def envelope(y , rate, threshold):
    mask=[]
    y=pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10) ,  min_periods=1 , center = True).mean()
    for mean in y_mean:
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


# In[10]:


#The clean Audio Files are redirected to Clean Audio Folder Directory 
import glob,pickle
for file in tqdm(glob.glob(r'D:\Minor Project\speech-emotion-recognition-ravdess-data\\**\\*.wav')):
    file_name = os.path.basename(file)
    signal , rate = librosa.load(file, sr=16000)
    mask = envelope(signal,rate, 0.0005)
    wavfile.write(filename= r'D:\Minor Project\clean_speech\\'+str(file_name), rate=rate,data=signal[mask])


# ### STEP 7 - FEATURE EXTRACTION
# 
# Define a function extract_feature to extract the mfcc, chroma, and mel features from a sound file. This function takes 4 parameters- the file name and three Boolean parameters for the three features:
# 
# **mfcc:** Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound
# 
# **chroma:** Pertains to the 12 different pitch classes
# 
# **mel: Mel Spectrogram Frequency**

# In[11]:


#Feature Extraction of Audio Files Function 
#Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result


# ### STEP 8 - LABELS CLASSIFICATION
# 
# Now, let’s define a dictionary to hold numbers and the emotions available in the RAVDESS dataset, and a list to hold those we want- calm, happy, fearful, disgust.

# In[12]:


#Emotions in the RAVDESS dataset to be classified Audio Files based on . 
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
#These are the emotions User wants to observe more :
observed_emotions=['calm', 'happy', 'fearful', 'disgust']


# ### STEP 9 - LOADING OF DATA & SPLITTING OF DATASET
# 
# Now, let’s load the clean data with a function load_data() – this takes in the relative size of the test set as parameter. x and y are empty lists; we’ll use the glob() function from the glob module to get all the pathnames for the sound files in our dataset,

# In[13]:


#Load the clean data
from glob import glob
import os
import glob
def load_data(test_size=0.33):
    x,y=[],[]
    answer = 0
    for file in glob.glob(r'D:\Minor Project\clean_speech\\*.wav'):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            answer += 1
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append([emotion,file_name])
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


# Time to split the dataset into training and testing sets! Let’s keep the test set 25% of everything and use the load_data function for this.

# In[14]:


#Split the dataset
import librosa
import numpy as np
x_train,x_test,y_trai,y_tes=load_data(test_size=0.25)
print(np.shape(x_train),np.shape(x_test), np.shape(y_trai),np.shape(y_tes))
y_test_map = np.array(y_tes).T
y_test = y_test_map[0]
test_filename = y_test_map[1]
y_train_map = np.array(y_trai).T
y_train = y_train_map[0]
train_filename = y_train_map[1]
print(np.shape(y_train),np.shape(y_test))
print(*test_filename,sep="\n")


# In[15]:


#Get the shape of the training and testing datasets
# print((x_train.shape[0], x_test.shape[0]))
print((x_train[0], x_test[0]))
#Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')


# ### STEP 10 - APPLY MLP CLASSIFIER
# 
# Now, let’s Apply a MLPClassifier. This is a Multi-layer Perceptron Classifier; it optimizes the log-loss function using LBFGS or stochastic gradient descent. Unlike SVM or Naive Bayes, the MLPClassifier has an internal neural network for the purpose of classification. This is a feedforward ANN model.

# In[16]:


# Apply Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)


# In[17]:


#Train the model
model.fit(x_train,y_train)


# ### STEP 11 - SAVING THE MODEL

# In[18]:


#SAVING THE MODEL
import pickle
# Save the Modle to file in the current working directory
#For any new testing data other than the data in dataset

Pkl_Filename = "Speech_Emotions_Recognition_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)


# ### STEP 11 - LOAD THE SAVED MODEL

# In[19]:


# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Speech_Emotions_Recognition_Model = pickle.load(file)

Speech_Emotions_Recognition_Model


# ### STEP 12 - PREDICT THE TEST DATA USING THE SAVED MODEL
# 
# Let’s predict the values for the test set from saved model. This gives us y_pred (the predicted emotions for the features in the test set).

# In[20]:


#predicting :
y_pred=Speech_Emotions_Recognition_Model.predict(x_test)
y_pred


# ### STEP 13 - SUMMARIZATION OF PREDICTED DATA
# 
# To calculate the accuracy of our model, we’ll call up the accuracy_score() function we imported from sklearn. Finally, we’ll round the accuracy to 2 decimal places and print it out.

# In[21]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

results = confusion_matrix(y_test, y_pred)

print('Confusion Matrix')
print(results)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Report")
print(classification_report(y_test, y_pred))


# In[22]:


#Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))


# ### STEP 14 - STORE THE PREDICTED FILE IN .CSV FILE

# In[23]:


#Store the Prediction probabilities into CSV file 
import numpy as np
import pandas as pd
y_pred1 = pd.DataFrame(y_pred, columns=['predictions'])
y_pred1['file_names'] = test_filename
print(y_pred1)
y_pred1.to_csv('predictionfinal.csv')


# # END

# In[ ]:




