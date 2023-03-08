from flask import Flask, request, jsonify
from tempfile import NamedTemporaryFile
import librosa
import numpy as np
import pydub
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

clf = joblib.load('mlp_classifier.joblib')
scaler = joblib.load('scaler.joblib')
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello Flask Here"

@app.route('/predict', methods = ['POST'])
def predict():
    audio = request.files['audio_file']
    with NamedTemporaryFile(delete=False) as temp:
        audio.save(temp.name)
        y, sr = librosa.load(temp.name, sr=None)
        features = extract_feature(temp.name, True, True, True)
    X = scaler.transform([features])
    emotion_labels = ['angry', 'sad', 'neutral', 'happy']
    proba = clf.predict_proba(X)[0]
    emotion_idx = np.argmax(proba)
    predicted_emotion = emotion_labels[emotion_idx]
    return jsonify({'Emotion': predicted_emotion})

# def predict():
#     audio = request.files['audio_file']
#     y, sr = librosa.load(audio, sr=None)
#     features = extract_feature(audio, True, True, True)
#     X = scaler.transform([features])
#     emotion_labels = ['angry', 'sad', 'neutral', 'happy']
#     proba = clf.predict_proba(X)[0]
#     emotion_idx = np.argmax(proba)
#     predicted_emotion = emotion_labels[emotion_idx]
#     return jsonify({'Emotion': predicted_emotion})

def extract_feature(file_name, mfcc, chroma, mel):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    if chroma:
        stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

# def extract_feature(file_name, mfcc, chroma, mel):
#     audio = pydub.AudioSegment.from_file(file_name)
#     audio = audio.set_channels(1)
#     audio = audio.set_frame_rate(22050)
#     X = np.array([])
#     sample_rate = audio.frame_rate
#     if chroma:
#         stft = np.abs(librosa.stft(audio.get_array_of_samples()))
#         result = np.array([])
#         chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
#         result = np.concatenate((result, np.mean(chroma)))
#     if mfcc:
#         mfccs = librosa.feature.mfcc(y=audio.get_array_of_samples(), sr=sample_rate, n_mfcc=40)
#         result = np.concatenate((result, np.mean(mfccs)))
#     if mel:
#         mel = librosa.feature.melspectrogram(audio.get_array_of_samples(), sr=sample_rate, n_mels=128)
#         result = np.concatenate((result, np.mean(mel)))
#     X = np.append(X, result)
#     return X


# def predictEmotion():
#     # Load the pre-trained MLP classifier and StandardScaler object
#     clf = joblib.load('mlp_classifier.joblib')
#     scaler = joblib.load('scaler.joblib')
#
#     # Load the single audio file and extract features
#     audio_file = request.form.get('audio_file')
#     y, sr = librosa.load(audio_file, sr=None)
#     features = extract_feature(audio_file, True, True, True)
#
#     # Normalize the extracted features
#     X = scaler.transform([features])
#
#     # Make a prediction on the normalized features
#     emotion_labels = ['angry', 'sad', 'neutral', 'happy']
#     proba = clf.predict_proba(X)[0]
#     emotion_idx = np.argmax(proba)
#     predicted_emotion = emotion_labels[emotion_idx]
#
#     return jsonify({'Emotion': predicted_emotion})

if __name__ == '__main__':
    app.run(debug=True)