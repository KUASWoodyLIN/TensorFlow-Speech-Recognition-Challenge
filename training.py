import os
import re
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

from sklearn.cross_validation import train_test_split

from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.layers import *
from keras.callbacks import *
from keras.utils.np_utils import to_categorical


path, _ = os.path.split(os.path.abspath(__file__))
DATA_DIR = path + '/train_data/train/audio/'
# DATA_DIR = path + '/train_data/train/small_audio/'    # small_audio, single_data
ALL_LABELS = 'yes bird happy five eight left house one four six two marvin nine dog seven stop no go ' \
             'right sheila zero cat on wow off down up _background_noise_ _silence_ three bed tree'.split()
POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}
len(id2name)


def load_data(data_dir):
  sample = {}
  for key in POSSIBLE_LABELS:
    sample[key] = []
  sample['silence'] = []
  sample['unknown'] = []

  all_files = glob(os.path.join(data_dir, '*/*wav'))
  for files in all_files:
    _, dir = files.split(DATA_DIR)
    label, file = dir.split('/')
    if label in POSSIBLE_LABELS:
      sample[label].append(files)
    elif label in ['_background_noise_', '_silence_']:
      sample['silence'].append(files)
    elif label not in POSSIBLE_LABELS:
      sample['unknown'].append(files)
    else:
      print('error')

  x_train, x_test, y_train, y_test = np.array([]), np.array([]), np.array([]), np.array([])
  for key in sample.keys():
    y = [name2id[key] for i in range(len(sample[key]))]
    xtr, xtt, ytr, ytt = train_test_split(sample[key], y, test_size=0.1)
    x_train = np.hstack((x_train, xtr))
    x_test = np.hstack((x_test, xtt))
    y_train = np.hstack((y_train, ytr))
    y_test = np.hstack((y_test, ytt))

  # random data
  indices_train = np.random.permutation(len(x_train))
  indices_test = np.random.permutation(len(x_test))

  rand_x_train = x_train[indices_train]
  rand_x_test = x_test[indices_test]
  rand_y_train = y_train[indices_train]
  rand_y_test = y_test[indices_test]

  rand_y_train = to_categorical(rand_y_train, num_classes=len(POSSIBLE_LABELS))
  rand_y_test = to_categorical(rand_y_test, num_classes=len(POSSIBLE_LABELS))

  print('There are {} train and {} val samples'.format(len(rand_x_train), len(rand_x_test)))
  return rand_x_train, rand_x_test, rand_y_train, rand_y_test, sample['silence']


# def load_silence():



def read_wav_file(file_abs):
  print(file_abs)
  sample_rate, samples = wavfile.read(file_abs)
  # normalization
  samples = samples.astype(np.float32) / np.iinfo(np.int16).max
  return samples


def process_wav_file(data, silence_data=np.array([]), window_size=20, step_size=10, eps=1e-10):
  # 1 sec
  L = 16000
  nperseg = int(round(window_size * L / 1e3))
  noverlap = int(round(step_size * L / 1e3))
  spectrogram = []
  for single_data in data:
    if len(single_data) > L:
      i = np.random.randint(0, len(single_data) - L)
      single_data = single_data[i:(i+L)]
    elif len(single_data) < L:
      # no silence way
      print(silence_data.any())
      if silence_data.any():
        rem_len = L - len(single_data)
        print(len(silence_data) - rem_len)
        i = np.random.randint(0, len(silence_data) - rem_len)
        silence_part = silence_data[i:(i + L)]
        j = np.random.randint(0, rem_len)
        silence_part_left = silence_part[0:j]
        silence_part_right = silence_part[j:rem_len]
        single_data = np.concatenate([silence_part_left, single_data, silence_part_right])
      else:
        single_data = np.concatenate([single_data, np.zeros(L - len(single_data))])

    freqs, time, spec = signal.spectrogram(single_data,
                                            fs=L,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)

    spectrogram.append(np.log(spec.astype(np.float32) + eps))
  spectrogram = np.expand_dims(np.array(spectrogram), 3)

  return freqs, time, spectrogram


x_train, x_test, y_train, y_test, silence_files = load_data(DATA_DIR)
silence_data = np.concatenate([read_wav_file(x) for x in silence_files])
x_train = [read_wav_file(x) for x in x_train]
x_test = [read_wav_file(x) for x in x_test]
freqs, times, x_train = process_wav_file(x_train, silence_data)
freqs, times, x_test = process_wav_file(x_test)

print(x_train.shape[1:])
x_in = Input(shape=x_train.shape[1:])
x = BatchNormalization()(x_in)
for i in range(4):
    x = Conv2D(16*(2 ** i), (3,3))(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (1, 1))(x)
x_branch_1 = GlobalAveragePooling2D()(x)
x_branch_2 = GlobalMaxPool2D()(x)
x = concatenate([x_branch_1, x_branch_2])
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(len(POSSIBLE_LABELS), activation='softmax')(x)
model = Model(inputs=x_in, outputs=x)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
              metrics=['accuracy'])

epochs = 150
batch_size = 64
file_name = str(epochs) + '_' + str(batch_size)
cbks = [
    EarlyStopping(monitor='val_loss',
                  patience=5,
                  verbose=1,
                  min_delta=0.01,
                  mode='min'),
    ReduceLROnPlateau(monitor='val_loss',
                      factor=0.1,
                      patience=3,
                      verbose=1,
                      epsilon=0.01,
                      mode='min'),
    TensorBoard(log_dir='logs/' + file_name)
]
print(x_train.shape)
model.fit(x_train, y_train, epochs=epochs,
          batch_size=batch_size,
          shuffle=True, verbose=1,
          validation_data=(x_test, y_test),
          callbacks=cbks)

model.save('h5/' + file_name + '.h5')
score = model.evaluate(x_test, y_test, verbose=0)

print('test cost:', score)

