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
  all_files_name = [i.split('/audio/', 1)[1] for i in all_files_abs]
  return all_files_abs, all_files_name


def read_wav_file(fname):
  print(fname)
  sample_rate, samples = wavfile.read(fname)
  samples = samples.astype(np.float32) / np.iinfo(np.int16).max
  return samples


def process_wav_file(fname, window_size=20, step_size=10, eps=1e-10):
  wav = read_wav_file(fname)

  # 1 sec
  L = 16000
  nperseg = int(round(window_size * L / 1e3))
  noverlap = int(round(step_size * L / 1e3))
  if len(wav) > L:
    i = np.random.randint(0, len(wav) - L)
    wav = wav[i:(i+L)]
  elif len(wav) < L:
    wav = np.concatenate([wav, np.zeros(L - len(wav))])

  freqs, time, spec = signal.spectrogram(wav,
                                          fs=L,
                                          window='hann',
                                          nperseg=nperseg,
                                          noverlap=noverlap,
                                          detrend=False)

  spec = np.log(spec.astype(np.float32) + eps)
  return spec


def test_generator(test_batch_size, test_paths):
  while True:
    for start in range(0, len(test_paths), test_batch_size):
      x_batch = []
      end = min(start + test_batch_size, len(test_paths))
      this_paths = test_paths[start:end]
      for x in this_paths:
        x_batch.append(process_wav_file(x))
      x_batch = np.expand_dims(np.array(x_batch), 3)
      yield x_batch


def transform(listdir, label, size):
  label_str = []
  for i in range(size):
      temp = listdir[label[i]]
      label_str.append(temp)
  return label_str


all_files_abs, all_files_name = load_wav(DATA_DIR)

model = load_model('h5/150_64.h5')
predict = model.predict_generator(test_generator(64, all_files_abs), int(np.ceil(len(all_files_abs)/64)))
predict = np.argmax(predict, axis=1)
label_str = transform(id2name, predict, len(all_files_abs))

with open('price_pred.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(['fname', 'label'])
    for i in range(len(label_str)):
        w.writerow([all_files_name[i], label_str[i]])
