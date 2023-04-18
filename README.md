# Speech-Emotion-Recognition
Speech Emotion Recognition Python Project using ML and DL frameworks.

The project is about multi class classification, that aims of predicting an emotion 
via the learning of the caracteristics of a speech signal in a form of an audio file (format '.wav').

In this project:

-The Dataset used is the RAVDESS Emotional speech audio downloaded from Kaggle. 
This dataset have 24 folders.
Each folder contains 60 audio files recorded by an different actor.
Hence, total of 60*24 i.e 1440 audio files.
Actors are 12 males and 12 females.
The dataset consists a total of 8 emotions, or 8 classes.
The emotions are: "Sad", "Happy", "Angry", "disgust", "Neutral", "Surprised", "Calm" and "Fearful".
The size of the dataset is 450mbs.

-The major libraries used in my notebook are:\n
1-Librosa: For speech feature extraction from raw audio files.\n
2-Scikitlearn: For ML and Preprocessing\n
3-Tensorflow: For building Deep Learning Models
And of course, Numpy, Pandas and matplotlib among others.

-The notebook consists of several major parts:
1-Data Preparation: where I extract data from different folders, as well as features with proper techniques and then store them into a DataFrame.
2-EDA: where I explore some statistics about my data and display a sample for each emotion.
3-Deep Learning Section: consists for building and traning 3 different architectures of an ANN.
The first model is a CNN-LSTM (particulary Bidirectional).
The second one is a simple LSTM model.
The final one is an improved version of the last one.
The section display also some statistics about the training and gives accurate evaluation of each model as classifiers.
4-Traditional ML models: divided into Supervised and Unsupervised.
5-Two untried models and a conclusion.
