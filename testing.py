import os
import csv
import numpy as np
from scipy.io import wavfile
from scipy import signal
from glob import glob
from keras.models import load_model


path, _ = os.path.split(os.path.abspath(__file__))
DATA_DIR = path + '/test/audio/'
# DATA_DIR = path + '/test/small/'
POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}
len(id2name)


def load_wav(data_dir):
  all_files_abs = glob(os.path.join(data_dir, '*wav'))
  file_name = [i.split('/audio/', 1)[1] for i in all_files_abs]
  samples_list = []
  for i in range(len(all_files_abs)):
    print(data_dir + file_name[i])
    sample_rate, samples = wavfile.read(data_dir + file_name[i])
    samples = samples.astype(np.float32) / np.iinfo(np.int16).max
    samples_list.append(samples)
  return samples_list, file_name


def process_wav_file(data, window_size=20, step_size=10, eps=1e-10):
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
      single_data = np.concatenate([single_data, np.zeros(L - len(single_data))])

    freqs, time, spec = signal.spectrogram(single_data,
                                            fs=L,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)

    spectrogram.append(np.log(spec.astype(np.float32) + eps))
  spectrogram = np.expand_dims(np.array(spectrogram), 3)

  return spectrogram


def transform(listdir, label, size):
  label_str = []
  for i in range(size):
      temp = listdir[label[i]]
      label_str.append(temp)
  return label_str


wav_data, file_name = load_wav(DATA_DIR)

split_piece = 6
wav_size = len(wav_data) / split_piece
x_test = []
for i in range(int(wav_size)+1):
  if i <= int(wav_size):
    print(wav_data[i*split_piece:(i+1)*split_piece])
    spectrogram = process_wav_file(wav_data[i*split_piece:(i+1)*split_piece])
  else:
    print(wav_data[i*split_piece:len(wav_data)])
    spectrogram = process_wav_file(wav_data[i*split_piece:len(wav_data)])
  x_test.append(spectrogram)
# x_test = np.vstack((x_test))

model = load_model('h5/150_64.h5')
predict = [model.predict(x_test[i], verbose=1) for i in range(len(x_test))]
predict = np.vstack(predict)
predict = np.argmax(predict, axis=1)
label_str = transform(id2name, predict, len(wav_data))

with open('price_pred.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(['fname', 'label'])
    for i in range(len(label_str)):
        w.writerow([file_name[i], label_str[i]])
