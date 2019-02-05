from multiprocessing import Pool
import os

import librosa
import librosa.effects
import librosa.util
import numpy as np
import torch
from torch.utils.data import Dataset

# DEFAULT_DIR = '/home/kureta/Music/Billboard Hot 100 Singles Charts/' \
#               'Billboard Hot 100 Singles Chart (03.03.2018) Mp3 (320kbps) ' \
#               '[Hunter]/Billboard Hot 100 Singles Chart (03.03.2018)'
DEFAULT_DIR = '/home/kureta/Music/misc/'
# DEFAULT_DIR = '/home/kureta/Music/bach complete/Bach 2000 v01CD01 (Cantatas BWV 01-03)/'
FRAME_LENGTH = 1024
HOP_LENGTH = 512


def do_multiprocess(function, args_list, num_processes=8):
    with Pool(num_processes) as p:
        results = list(p.map(function, args_list))
    return results


def load_audio_file(path, sr=44100, mono=True):
    audio, _ = librosa.load(path, sr=sr, mono=mono)
    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio, top_db=60, ref=np.max, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    return audio


def get_stft(audio):
    return librosa.stft(audio, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)


def recursive_file_paths(directory):
    file_paths = []
    for dirname, dirnames, filenames in os.walk(directory):
        # print path to all filenames.
        for filename in filenames:
            if filename.endswith(".mp3"):
                file_paths.append(os.path.join(dirname, filename))

    return file_paths


class AtemporalDataset(Dataset):
    def __init__(self, audio_directory=DEFAULT_DIR, normalize=True):
        super(AtemporalDataset, self).__init__()

        file_paths = recursive_file_paths(audio_directory)
        audio_list = do_multiprocess(load_audio_file, file_paths)
        del file_paths
        stft_list = do_multiprocess(get_stft, audio_list)
        del audio_list
        complex_stfts = np.concatenate(stft_list, axis=1)
        del stft_list
        self.spectra = torch.from_numpy(np.abs(complex_stfts)).transpose(1, 0)
        del complex_stfts

        if normalize:
            self.spectra /= self.spectra.max()

    def __len__(self):
        return self.spectra.shape[0]

    def __getitem__(self, index):
        return self.spectra[index]


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
        self.encoder.eval()
        with torch.no_grad():
            x = self.encoder(self.absoulte_examples[index].transpose(0, 1)).transpose(0, 1)
        return x
