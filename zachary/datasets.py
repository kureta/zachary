import os
from multiprocessing import Pool

import torch
from torch.utils.data import Dataset

import numpy as np

from librosa import load
from librosa import stft, istft
from librosa.effects import trim
from librosa.util import normalize

# DEFAULT_DIR = "/home/kureta/Music/Billboard Hot 100 Singles Charts/" \
#               "Billboard Hot 100 Singles Chart (03.03.2018) Mp3 (320kbps) [Hunter]"
DEFAULT_DIR = "/home/kureta/Music/bach complete/Bach 2000 v01CD01 (Cantatas BWV 01-03)/"
FRAME_LENGTH = 1024
HOP_LENGTH = 512


def recursive_file_paths(directory):
    file_paths = []
    for dirname, dirnames, filenames in os.walk(directory):
        # print path to all filenames.
        for filename in filenames:
            if filename.endswith(".mp3"):
                file_paths.append(os.path.join(dirname, filename))

    return file_paths


def complex_stft(x):
    c_stft = stft(x, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH, center=False, pad_mode='none')
    magnitude = np.abs(c_stft)
    phase = np.angle(c_stft)

    # dims = (channels, duration, mag/phase)
    return np.concatenate((np.expand_dims(magnitude, 2), np.expand_dims(phase, 2)), axis=2)


def istft_(x):
    return istft(x[:, :, 0] * np.exp(1j * x[:, :, 1]), hop_length=512, win_length=1024, center=False)


def load_file(path):
    audio, _ = load(path, sr=44100, mono=True)
    audio, _ = trim(audio, top_db=40, ref=np.max, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    audio = normalize(audio)
    audio = complex_stft(audio)

    return audio


class MagPhaseSTFT(Dataset):
    def __init__(self, file_paths=None, example_length=3, transform=None, target_transform=None):
        super(MagPhaseSTFT, self).__init__()
        self.transform = transform
        self.target_transform = target_transform

        if file_paths is None:
            file_paths = recursive_file_paths(DEFAULT_DIR)
        with Pool(8) as p:
            results = list(p.map(load_file, file_paths))

        # dims = (channels, duration, mag/phase)
        self.audio = torch.from_numpy(np.concatenate(results, 1))
        self.mins = self.audio.min(dim=1, keepdim=True)[0]
        self.maxes = self.audio.max(dim=1, keepdim=True)[0]
        self.audio = (self.audio - self.mins) / (self.maxes - self.mins)

        self.example_length = example_length

    def denormalize(self, x):
        return x * (self.maxes - self.mins) + self.mins

    def update_strided_views(self):
        self.num_frames = self.audio.shape[1]
        self.num_examples = self.num_frames - self.example_length + 1

        strides = self.audio.stride()
        self.examples = self.audio.as_strided((self.num_examples, HOP_LENGTH + 1, self.example_length, 2),
                                              (strides[1], strides[0], strides[1], strides[2]))

    @property
    def example_length(self):
        return self._example_length

    @example_length.setter
    def example_length(self, value):
        self._example_length = value
        self.update_strided_views()

    def __len__(self):
        return self.examples.shape[0]

    def __getitem__(self, index):
        result = self.examples[index]
        if self.transform is not None:
            example = self.transform(result)
        else:
            example = result

        if self.target_transform is not None:
            target = self.target_transform(result)
        else:
            target = result
        return example, target
