from dataclasses import dataclass

import librosa


@dataclass(frozen=True)
class Configuration:
    sample_rate: int = 44100
    frame_length: int = 1024
    hop_length: int = 512
    lowest_note: str = 'e1'
    highest_note: str = 'a5'
    silence_threshold: float = 36.  # db
    audio_dir: str = '/home/kureta/Music/chorales/'

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
