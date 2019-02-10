from dataclasses import dataclass
from typing import List

import librosa


@dataclass(frozen=True)
class Configuration:
    sample_rate: int = 44100
    frame_length: int = 1024
    hop_length: int = 512
    lowest_note: str = 'e1'
    highest_note: str = 'a5'
    max_freq: float = librosa.note_to_hz(highest_note)
    min_freq: float = librosa.note_to_hz(lowest_note)
    pitch_range: int = librosa.note_to_midi(highest_note) - librosa.note_to_midi(lowest_note) + 1
    silence_threshold: float = 36.  # db
    default_dir: str = '/home/kureta/Music/chorales/'
    __fft_frequencies: List[float] = librosa.fft_frequencies(sr=44100, n_fft=1024)

    def fft_bin_to_freq(self, x):
        return self.__fft_frequencies[x]

    def frames_to_duration(self, num_frames):
        return librosa.frames_to_time(num_frames, sr=self.sample_rate,
                                      hop_length=self.hop_length, n_fft=self.frame_length)
