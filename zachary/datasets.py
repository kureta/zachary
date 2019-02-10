import os
from functools import partial
from multiprocessing import Pool

import librosa
import librosa.effects
import librosa.util
import numpy as np
import torch
from torch.utils.data import Dataset

from zachary.constants import Configuration
from zachary.feature_extraction import get_features_from_signal


def do_multiprocess(function, args_list, num_processes=8):
    with Pool(num_processes) as p:
        results = list(p.map(function, args_list))
    return results


def recursive_file_paths(directory):
    file_paths = []
    for dirname, dirnames, filenames in os.walk(directory):
        # print path to all filenames.
        for filename in filenames:
            if filename.endswith(".mp3"):
                file_paths.append(os.path.join(dirname, filename))

    return file_paths


class AtemporalDataset(Dataset):
    def __init__(self, conf=Configuration()):
        super(AtemporalDataset, self).__init__()
        self.conf = conf

        file_paths = recursive_file_paths(self.conf.default_dir)
        audio_list = do_multiprocess(self.load_audio_file, file_paths)
        del file_paths

        ex = partial(get_features_from_signal, conf=self.conf)
        features_list = do_multiprocess(ex, audio_list)
        spectra, pitches, confidences, loudnesses = zip(*features_list)
        del audio_list, features_list

        self.spectra = torch.from_numpy(np.concatenate(spectra, axis=0))
        self.pitches = torch.from_numpy(np.concatenate(pitches))
        self.confidences = torch.from_numpy(np.concatenate(confidences))
        self.loudnesses = torch.from_numpy(np.concatenate(loudnesses))
        del spectra, pitches, confidences, loudnesses

        # TODO: convert feature tensors into indices. Handle -inf pitches, normalize confidences and loudnesses.

        self.maxima = self.spectra.max(0)[0]

    def load_audio_file(self, path):
        audio, _ = librosa.load(path, sr=self.conf.sample_rate)
        audio = librosa.util.normalize(audio)
        audio, _ = librosa.effects.trim(audio, top_db=self.conf.silence_threshold,
                                        frame_length=self.conf.frame_length, hop_length=self.conf.hop_length)
        return audio

    def __len__(self):
        return self.spectra.shape[0]

    def __getitem__(self, index):
        return self.spectra[index] / self.maxima, \
               self.pitches[index], \
               self.confidences[index], \
               self.loudnesses[index]


class GANDataset(Dataset):
    def __init__(self, atemporal_dataset, encoder, example_length=64, stft_hop_length=32):
        super(GANDataset, self).__init__()

        self._example_length = example_length
        self._stft_hop_length = stft_hop_length

        self.spectra = atemporal_dataset.spectra
        self.encoder = encoder
        self.absoulte_examples = None
        self.update_strided()

    def update_strided(self):
        stride = self.spectra.stride()
        shape = self.spectra.shape
        n_examples = (shape[0] - self.example_length) // self.stft_hop_length + 1
        self.absoulte_examples = self.spectra.as_strided(
            (n_examples, shape[1], self.example_length),
            (stride[0] * self.stft_hop_length, stride[1], stride[0]))

    @property
    def example_length(self):
        return self._example_length

    @example_length.setter
    def example_length(self, value):
        self._example_length = value
        self.update_strided()

    @property
    def stft_hop_length(self):
        return self._stft_hop_length

    @stft_hop_length.setter
    def stft_hop_length(self, value):
        self._stft_hop_length = value
        self.update_strided()

    def __len__(self):
        return self.absoulte_examples.shape[0]

    def __getitem__(self, index):
        return self.absoulte_examples[index]
