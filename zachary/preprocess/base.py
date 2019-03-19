from dataclasses import dataclass
from functools import partial

import librosa
import numpy as np
import torch
from pretty_midi import PrettyMIDI
from torch.utils.data import Dataset

from zachary.preprocess.utils import recursive_file_paths, do_multiprocess, load_audio_file, spectrum_from_signal


@dataclass(frozen=True)
class Configuration:
    sample_rate: int = 44100
    frame_length: int = 1024
    hop_length: int = 512
    lowest_note: str = 'e1'
    highest_note: str = 'a5'
    silence_threshold: float = 36.  # db
    audio_dir: str = '/home/kureta/Music/Palestrina - Missa Papæ Marcelli - Ensemble Officium, Wilfried Rombach/'
    midi_dir: str = '/home/kureta/Music/midi/palestrina/'

    @property
    def max_hz(self):
        return librosa.note_to_hz(self.highest_note)

    @property
    def min_hz(self):
        return librosa.note_to_hz(self.lowest_note)

    @property
    def max_midi(self):
        return librosa.note_to_midi(self.highest_note)

    @property
    def min_midi(self):
        return librosa.note_to_midi(self.lowest_note)

    @property
    def pitch_range(self):
        return librosa.note_to_midi(self.highest_note) - librosa.note_to_midi(self.lowest_note) + 1

    def bin_to_hz(self, x):
        return librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)[x]

    def frames_to_time(self, num_frames):
        return librosa.frames_to_time(num_frames, sr=self.sample_rate,
                                      hop_length=self.hop_length, n_fft=self.frame_length)

    def time_to_frames(self, seconds):
        return librosa.time_to_frames(seconds, sr=self.sample_rate,
                                      hop_length=self.hop_length, n_fft=self.frame_length)

    def samples_to_time(self, num_samples):
        return librosa.samples_to_time(num_samples, sr=self.sample_rate)

    def time_to_samples(self, seconds):
        return librosa.time_to_samples(seconds, sr=self.sample_rate)

    def frames_to_samples(self, num_frames):
        return librosa.frames_to_samples(num_frames, hop_length=self.hop_length, n_fft=self.frame_length)

    def samples_to_frames(self, num_samples):
        return librosa.samples_to_frames(num_samples, hop_length=self.hop_length, n_fft=self.frame_length)


class BaseDataset(Dataset):
    def __init__(self, conf=Configuration()):
        super(Dataset, self).__init__()
        self.conf = conf

        file_paths = recursive_file_paths(self.conf.audio_dir)
        signals = do_multiprocess(partial(load_audio_file, conf=conf), file_paths)
        del file_paths

        spectra = do_multiprocess(partial(spectrum_from_signal, conf=conf), signals)
        del signals

        spectra = do_multiprocess(librosa.amplitude_to_db, spectra)

        self.spectra = torch.from_numpy(np.concatenate(spectra, axis=0))
        del spectra

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class Midi(PrettyMIDI):
    def get_piano_roll(self, fs=100, times=None):
        # If there are no instruments, return an empty array
        if len(self.instruments) == 0:
            return np.zeros((128, 0))

        # Get piano rolls for each instrument
        piano_rolls = [i.get_piano_roll(fs=fs, times=times)
                       for i in self.instruments]
        # Allocate piano roll,
        # number of columns is max of # of columns in all piano rolls
        piano_roll = np.zeros((128, np.max([p.shape[1] for p in piano_rolls])))
        # Sum each piano roll into the aggregate piano roll
        for roll in piano_rolls:
            piano_roll[:, :roll.shape[1]] = np.maximum(piano_roll[:, :roll.shape[1]], roll)
        return piano_roll


def load_midi_file(path, conf):
    midi = Midi(path)
    return midi.get_piano_roll(
        fs=librosa.time_to_frames(100, conf.sample_rate, conf.hop_length, conf.frame_length) / 100
    )


def trim_zeros(x):
    # Trims only from the beginning.
    s = np.sum(np.abs(x), axis=0)
    first = (s != 0).argmax(axis=0)
    return x[:, first:].T


class BaseMidiDataset(Dataset):
    def __init__(self, conf=Configuration()):
        super(Dataset, self).__init__()
        self.conf = conf

        file_paths = recursive_file_paths(self.conf.midi_dir, supported_extensions=['.mid'])
        matrices = do_multiprocess(partial(load_midi_file, conf=conf), file_paths)
        del file_paths

        trimmed = do_multiprocess(trim_zeros, matrices)
        del matrices

        self.midi = torch.from_numpy(np.concatenate(trimmed, axis=0).astype('float32'))
        del trimmed

        self.midi /= 127.

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
