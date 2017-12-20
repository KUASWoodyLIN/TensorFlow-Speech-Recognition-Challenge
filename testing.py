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
test_paths = glob(os.path.join(DATA_DIR, '*wav'))


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


def test_generator(test_batch_size):
  while True:
    print(range(0, len(test_paths), test_batch_size))
    for start in range(0, len(test_paths), test_batch_size):
      x_batch = []
      end = min(start + test_batch_size, len(test_paths))
      this_paths = test_paths[start:end]
      print("start", start, "end", end)
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


model = load_model('h5/150_64.hdf5')
print(int(np.ceil(len(test_paths)*1.0/64)))
predict = model.predict_generator(test_generator(64), int(np.ceil(len(test_paths)/64)))
predict = np.argmax(predict, axis=1)
label_str = transform(id2name, predict, len(test_paths))

submission = dict()
for i in range(len(test_paths)):
    fname, label = os.path.basename(test_paths[i]), id2name[predict[i]]
    submission[fname] = label

with open('starter_submission.csv', 'w') as fout:
  fout.write('fname,label\n')
  for fname, label in submission.items():
    fout.write('{},{}\n'.format(fname, label))

# with open('price_pred.csv', 'w') as f:
#     w = csv.writer(f)
#     w.writerow(['fname', 'label'])
#     for i in range(len(label_str)):
#         w.writerow([test_paths[i], label_str[i]])
