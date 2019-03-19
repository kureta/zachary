import os
from multiprocessing import Pool

import librosa
import librosa.effects
import librosa.util
import numpy as np


def do_multiprocess(function, args_list, num_processes=8):
    with Pool(num_processes) as p:
        results = list(p.map(function, args_list))
    return results


def recursive_file_paths(directory, supported_extensions=None):
    if supported_extensions is None:
        supported_extensions = ['.mp3', '.wav', '.flac', '.ape']

    file_paths = []
    for dirname, dirnames, filenames in os.walk(directory):
        # print path to all filenames.
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in supported_extensions:
                file_paths.append(os.path.join(dirname, filename))

    return file_paths


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def load_audio_file(path, conf):
    audio, _ = librosa.load(path, sr=conf.sample_rate)
    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio, top_db=conf.silence_threshold,
                                    frame_length=conf.frame_length, hop_length=conf.hop_length)
    return audio


def spectrum_from_signal(audio, conf):
    complex_stft = librosa.stft(audio, n_fft=conf.frame_length, hop_length=conf.hop_length,
                                win_length=conf.frame_length, window='hann', center=True, pad_mode='constant')

    return np.abs(complex_stft.T).astype('float32')
