import os
from glob import glob
import numpy as np
from scipy.io import wavfile
from scipy import signal

from sklearn.cross_validation import train_test_split

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import *
from keras.callbacks import *
from keras.utils.np_utils import to_categorical


path, _ = os.path.split(os.path.abspath(__file__))
DATA_DIR = path + '/train/audio/'
# DATA_DIR = path + '/train/small/'    # small_audio, single_data
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

  x_train, x_valid, y_train, y_valid = np.array([]), np.array([]), np.array([]), np.array([])
  for key in sample.keys():
    y = [name2id[key] for i in range(len(sample[key]))]
    xtr, xtt, ytr, ytt = train_test_split(sample[key], y, test_size=0.1)
    x_train = np.hstack((x_train, xtr))
    x_valid = np.hstack((x_valid, xtt))
    y_train = np.hstack((y_train, ytr))
    y_valid = np.hstack((y_valid, ytt))

  # random data
  indices_train = np.random.permutation(len(x_train))
  indices_test = np.random.permutation(len(x_valid))

  rand_x_train = x_train[indices_train]
  rand_x_valid = x_valid[indices_test]
  rand_y_train = y_train[indices_train]
  rand_y_valid = y_valid[indices_test]

  rand_y_train = to_categorical(rand_y_train, num_classes=len(POSSIBLE_LABELS))
  rand_y_valid = to_categorical(rand_y_valid, num_classes=len(POSSIBLE_LABELS))

  print('There are {} train and {} val samples'.format(len(rand_x_train), len(rand_x_valid)))
  return rand_x_train, rand_x_valid, rand_y_train, rand_y_valid, sample['silence']


def read_wav_file(file_abs):
  # print(file_abs)
  sample_rate, samples = wavfile.read(file_abs)
  samples = samples.astype(np.float32) / np.iinfo(np.int16).max
  return samples


x_train, x_valid, y_train, y_valid, silence_files = load_data(DATA_DIR)
silence_data = np.concatenate([read_wav_file(x) for x in silence_files])


def process_wav_file(fname, window_size=20, step_size=10, eps=1e-10):
  wav = read_wav_file(fname)

  # 1 sec
  L = 16000
  nperseg = int(round(window_size * L / 1e3))
  noverlap = int(round(step_size * L / 1e3))
  if len(wav) > L:
    i = np.random.randint(0, len(wav) - L)
    wav = wav[i:(i + L)]
  elif len(wav) < L:
    rem_len = L - len(wav)
    i = np.random.randint(0, len(silence_data) - rem_len)
    silence_part = silence_data[i:(i + L)]
    j = np.random.randint(0, rem_len)
    silence_part_left = silence_part[0:j]
    silence_part_right = silence_part[j:rem_len]
    wav = np.concatenate([silence_part_left, wav, silence_part_right])

  freqs, time, spec = signal.spectrogram(wav,
                                          fs=L,
                                          window='hann',
                                          nperseg=nperseg,
                                          noverlap=noverlap,
                                          detrend=False)

  spec = np.log(spec.astype(np.float32) + eps)
  return spec


def train_generator(train_batch_size):
  while True:
    for start in range(0, len(x_train), train_batch_size):
      x_batch = []
      end = min(start + train_batch_size, len(x_train))
      i_train_batch = x_train[start:end]
      y_batch = y_train[start:end]
      for i in i_train_batch:
        x_batch.append(process_wav_file(i))
      x_batch = np.expand_dims(np.array(x_batch), 3)
      yield x_batch, y_batch


def valid_generator(val_batch_size):
  while True:
    for start in range(0, len(x_valid), val_batch_size):
      x_batch = []
      end = min(start + val_batch_size, len(x_valid))
      i_val_batch = x_valid[start:end]
      y_batch = y_valid[start:end]
      for i in i_val_batch:
        x_batch.append(process_wav_file(i))
      x_batch = np.expand_dims(np.array(x_batch), 3)
      yield x_batch, y_batch


x_in = Input(shape=(161, 99, 1))
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
callbacks = [
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
  ModelCheckpoint(monitor='val_loss',
                  filepath='h5/' + file_name + '.hdf5',
                  save_best_only=True,
                  mode='min'),
  TensorBoard(log_dir='logs/' + file_name)
]

history = model.fit_generator(generator=train_generator(64),
                              steps_per_epoch=344,
                              epochs=epochs,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=valid_generator(64),
                              validation_steps=int(np.ceil(len(x_valid)/64)))

# model.save('h5/' + file_name + '.h5')
# score = model.evaluate(x_valid, y_valid, verbose=0)

# print('test cost:', score)

