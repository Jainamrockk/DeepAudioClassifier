
---

# Capuchin Bird Call Deep Audio Classifier

This repository contains a deep learning model for classifying audio clips to detect Capuchin bird calls. The project utilizes TensorFlow and TensorFlow I/O for processing audio data and building the classifier.

## Features Used
Short-Time Fourier Transform (STFT): A technique to analyze the frequency content of audio signals over time, enabling the conversion of audio waves into spectrograms.
Resampling: Conversion of audio signals to a uniform sample rate (16 kHz) for consistent input to the model.
Data Augmentation: Techniques to enhance the training dataset and improve model robustness.
Convolutional Neural Networks (CNN): A deep learning architecture utilized for classifying spectrograms, consisting of convolutional and pooling layers.
TensorFlow Datasets: Efficient data loading and preprocessing pipelines to handle large audio datasets.
Model Evaluation Metrics: Precision, recall, and binary cross-entropy loss to evaluate model performance.


## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Audio Processing](#audio-processing)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Prediction](#prediction)
- [License](#license)

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Jainamrockk/DeepAudioClassifier.git
cd DeepAudioClassifier
pip install tensorflow tensorflow-gpu tensorflow_io matplotlib kaggle
```

## Dataset

The dataset used for this project is available on Kaggle: [Z by HP Unlocked Challenge 3: Signal Processing](https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing). Download and unzip the dataset:

```python
!kaggle datasets download -d kenjee/z-by-hp-unlocked-challenge-3-signal-processing
```

Unzip the downloaded file:

```python
import os
import zipfile

def unzip_file(zip_filepath, extract_to_dir):
    if not os.path.exists(extract_to_dir):
        os.makedirs(extract_to_dir)
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_to_dir)

zip_filepath = '/content/z-by-hp-unlocked-challenge-3-signal-processing.zip'
extract_to_dir = '/content/Data'
unzip_file(zip_filepath, extract_to_dir)
```

## Usage

### Audio Processing

Load and process the audio files to 16 kHz single-channel audio, then convert to spectrograms.

```python
import tensorflow as tf
import tensorflow_io as tfio

def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav
```

Convert the data to spectrograms:

```python
def preprocess(file_path, label):
    wave = load_wav_16k_mono(file_path)
    wav = wave[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label
```

### Model Architecture

The model consists of convolutional layers followed by dense layers to classify the spectrograms.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(1491,257,1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='Adam', loss='BinaryCrossentropy', metrics=['Recall', 'Precision'])
model.summary()
```

### Training

The model is trained using the processed dataset:

```python
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

train = data.take(36)
test = data.skip(36).take(15)

hist = model.fit(train, epochs=4, validation_data=test)
```

### Evaluation

The performance of the model is evaluated based on precision, recall, and loss metrics.

```python
import matplotlib.pyplot as plt

plt.title('Loss')
plt.plot(hist.history['loss'], 'r', label='Training Loss')
plt.plot(hist.history['val_loss'], 'b', label='Validation Loss')
plt.legend()
plt.show()

plt.title('Precision')
plt.plot(hist.history['precision'], 'r', label='Training Precision')
plt.plot(hist.history['val_precision'], 'b', label='Validation Precision')
plt.legend()
plt.show()

plt.title('Recall')
plt.plot(hist.history['recall'], 'r', label='Training Recall')
plt.plot(hist.history['val_recall'], 'b', label='Validation Recall')
plt.legend()
plt.show()
```
![alt text](https://github.com/Jainamrockk/DeepAudioClassifier/blob/main/Images/Loss.png)
![alt text](https://github.com/Jainamrockk/DeepAudioClassifier/blob/main/Images/Precision.png)
![alt text](https://github.com/Jainamrockk/DeepAudioClassifier/blob/main/Images/Recall.png)



### Prediction

The model can predict Capuchin bird calls in new audio clips and output results to a CSV file.

```python
import csv
from itertools import groupby

results = {}
for file in os.listdir(os.path.join('Data', 'Forest Recordings')):
    mp3 = os.path.join('Data', 'Forest Recordings', file)
    wav = load_mp3_16k_mono(mp3)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)
    yhat = model.predict(audio_slices)
    yhat = [1 if prediction > 0.99 else 0 for prediction in yhat]
    yhat = [key for key,group in groupby(yhat)]
    results[file] = yhat

postprocessed = {}
for file, scores in results.items():
    postprocessed[file] = tf.math.reduce_sum([key for key,group in groupby(scores)]).numpy()

with open('capuchin_scores.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['recording', 'capuchin_calls'])
    for key, value in postprocessed.items():
        writer.writerow([key, value])
```


