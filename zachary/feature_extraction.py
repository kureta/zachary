import essentia
import essentia.standard as es
import librosa


def get_features_from_signal(audio, conf):
    window = es.Windowing(type='hann')
    get_spectrum = es.Spectrum()
    get_melodia_pitches = es.PredominantPitchMelodia(maxFrequency=conf.max_freq, minFrequency=conf.min_freq,
                                                     frameSize=conf.frame_length, hopSize=conf.hop_length,
                                                     guessUnvoiced=False)
    pitch_filter = es.PitchFilter(useAbsolutePitchConfidence=False)
    eq_loudness = es.EqualLoudness(sampleRate=conf.sample_rate)
    get_loudness = es.Loudness()

    spectra = []
    loudnesses = []

    for frame in es.FrameGenerator(audio, frameSize=conf.frame_length, hopSize=conf.hop_length, startFromZero=True):
        windowed_frame = window(frame)
        spectra.append(get_spectrum(windowed_frame))
        loudnesses.append(get_loudness(windowed_frame))

    spectra = essentia.array(spectra)
    loudnesses = essentia.array(loudnesses)

    pitches, confidences = get_melodia_pitches(eq_loudness(audio))
    filtered_pitches = pitch_filter(pitches, confidences)

    return spectra, librosa.hz_to_midi(filtered_pitches[1:-1]), confidences[1:-1], loudnesses
