from multiprocessing import Pool

import librosa
import librosa.effects
import librosa.util
import numpy as np
import torch
from torch.utils.data import Dataset

# DEFAULT_DIR = "/home/kureta/Music/Billboard Hot 100 Singles Charts/" \
#               "Billboard Hot 100 Singles Chart (03.03.2018) Mp3 (320kbps) [Hunter]"
DEFAULT_DIR = '/home/kureta/Music/misc/'  # "/home/kureta/Music/bach complete/Bach 2000 v01CD01 (Cantatas BWV 01-03)/"
FRAME_LENGTH = 1024
HOP_LENGTH = 128


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


class SoundDataset(Dataset):
    def __init__(self, audio_directory=DEFAULT_DIR, example_length=15, stft_hop_length=5, normalize=True):
        super(SoundDataset, self).__init__()

        self._example_length = example_length
        self._stft_hop_length = stft_hop_length

        file_paths = librosa.util.find_files(audio_directory)
        audio_list = do_multiprocess(load_audio_file, file_paths)
        stft_list = do_multiprocess(get_stft, audio_list)

        complex_stfts = np.concatenate(stft_list, axis=1)
        shape = complex_stfts.shape
        tmp = np.zeros((*shape, 2), dtype='float32')
        tmp[:, :, 0] = complex_stfts.real
        tmp[:, :, 1] = complex_stfts.imag

        self.stfts = torch.from_numpy(tmp)
        self.spectra = torch.from_numpy(np.abs(complex_stfts))

        if normalize:
            self.stfts /= self.spectra.max()
            self.spectra /= self.spectra.max()

        self.complex_examples = None
        self.absoulte_examples = None

        self.update_strided()

    def update_strided(self):
        stride = self.stfts.stride()
        shape = self.stfts.shape
        n_examples = (shape[1] - self.example_length) // self.stft_hop_length + 1
        self.complex_examples = self.stfts.as_strided(
            (n_examples, shape[0], self.example_length, shape[2]),
            (stride[1] * self.stft_hop_length, stride[0], stride[1], stride[2]))

        stride = self.spectra.stride()
        shape = self.spectra.shape
        self.absoulte_examples = self.spectra.as_strided(
            (n_examples, shape[0], self.example_length),
            (stride[1] * self.stft_hop_length, stride[0], stride[1]))

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
        return self.complex_examples.shape[0]

    def __getitem__(self, index):
        return self.complex_examples[index], self.absoulte_examples[index]


class AtemporalDataset(Dataset):
    def __init__(self, audio_directory=DEFAULT_DIR, normalize=True):
        super(AtemporalDataset, self).__init__()

        file_paths = librosa.util.find_files(audio_directory)
        audio_list = do_multiprocess(load_audio_file, file_paths)
        stft_list = do_multiprocess(get_stft, audio_list)

        complex_stfts = np.concatenate(stft_list, axis=1)
        self.spectra = torch.from_numpy(np.abs(complex_stfts)).transpose(1, 0).contiguous()

        if normalize:
            self.spectra /= self.spectra.max()

    def __len__(self):
        return self.spectra.shape[0]

    def __getitem__(self, index):
        return self.spectra[index]
