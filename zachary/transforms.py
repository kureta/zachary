import torch
from torch.autograd import Variable
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


def _tlog10(x):
    return torch.log(x) / torch.log(x.new([10]))


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Scale(object):
    def __init__(self, factor=2 ** 31):
        self.factor = factor

    def __call__(self, tensor):
        if isinstance(tensor, (torch.LongTensor, torch.IntTensor)):
            tensor = tensor.float()

        return tensor / self.factor

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PadTrim(object):
    def __init__(self, max_len, fill_value=0):
        self.max_len = max_len
        self.fill_value = fill_value

    def __call__(self, tensor):
        if self.max_len > tensor.size(0):
            pad = torch.ones((self.max_len - tensor.size(0),
                              tensor.size(1))) * self.fill_value
            pad = pad.type_as(tensor)
            tensor = torch.cat((tensor, pad), dim=0)
        elif self.max_len < tensor.size(0):
            tensor = tensor[:self.max_len, :]
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(max_len={0})'.format(self.max_len)


class DownmixMono(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        if isinstance(tensor, (torch.LongTensor, torch.IntTensor)):
            tensor = tensor.float()

        if tensor.size(1) > 1:
            tensor = torch.mean(tensor.float(), 1, True)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'


class LC2CL(object):
    def __call__(self, tensor):
        return tensor.transpose(0, 1).contiguous()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SPECTROGRAM(object):
    def __init__(self, sr=16000, ws=400, hop=None, n_fft=None,
                 pad=0, window=torch.hann_window, wkwargs=None):
        if isinstance(window, Variable):
            self.window = window
        else:
            self.window = window(ws) if wkwargs is None else window(ws, **wkwargs)
            self.window = Variable(self.window, volatile=True)
        self.sr = sr
        self.ws = ws
        self.hop = hop if hop is not None else ws // 2
        # number of fft bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.n_fft = (n_fft - 1) * 2 if n_fft is not None else ws
        self.pad = pad
        self.wkwargs = wkwargs

    def __call__(self, sig):
        assert sig.dim() == 2

        if self.pad > 0:
            c, n = sig.size()
            new_sig = sig.new_empty(c, n + self.pad * 2)
            new_sig[:, :self.pad].zero_()
            new_sig[:, -self.pad:].zero_()
            new_sig.narrow(1, self.pad, n).copy_(sig)
            sig = new_sig

        spec_f = torch.stft(sig, self.n_fft, self.hop, self.ws,
                            self.window, center=False,
                            normalized=True, onesided=True).transpose(1, 2)
        spec_f /= self.window.pow(2).sum().sqrt()
        spec_f = spec_f.pow(2).sum(-1)  # get power of "complex" tensor (c, l, n_fft)
        return spec_f


class F2M(object):
    def __init__(self, n_mels=40, sr=16000, f_max=None, f_min=0.):
        self.n_mels = n_mels
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min

    def __call__(self, spec_f):
        n_fft = spec_f.size(2)

        m_min = 0. if self.f_min == 0 else 2595 * np.log10(1. + (self.f_min / 700))
        m_max = 2595 * np.log10(1. + (self.f_max / 700))

        m_pts = torch.linspace(m_min, m_max, self.n_mels + 2)
        f_pts = (700 * (10 ** (m_pts / 2595) - 1))

        bins = torch.floor(((n_fft - 1) * 2) * f_pts / self.sr).long()

        fb = torch.zeros(n_fft, self.n_mels)
        for m in range(1, self.n_mels + 1):
            f_m_minus = bins[m - 1].item()
            f_m = bins[m].item()
            f_m_plus = bins[m + 1].item()

            if f_m_minus != f_m:
                fb[f_m_minus:f_m, m - 1] = (torch.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
            if f_m != f_m_plus:
                fb[f_m:f_m_plus, m - 1] = (f_m_plus - torch.arange(f_m, f_m_plus)) / (f_m_plus - f_m)

        fb = Variable(fb)
        spec_m = torch.matmul(spec_f, fb)  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)
        return spec_m


class SPEC2DB(object):
    def __init__(self, stype="power", top_db=None):
        self.stype = stype
        self.top_db = -top_db if top_db > 0 else top_db
        self.multiplier = 10. if stype == "power" else 20.

    def __call__(self, spec):
        spec_db = self.multiplier * _tlog10(spec / spec.max())  # power -> dB
        if self.top_db is not None:
            spec_db = torch.max(spec_db, spec_db.new([self.top_db]))
        return spec_db


class MEL2(object):
    def __init__(self, sr=16000, ws=400, hop=None, n_fft=None,
                 pad=0, n_mels=40, window=torch.hann_window, wkwargs=None):
        self.window = window(ws) if wkwargs is None else window(ws, **wkwargs)
        self.window = Variable(self.window, requires_grad=False)
        self.sr = sr
        self.ws = ws
        self.hop = hop if hop is not None else ws // 2
        self.n_fft = n_fft  # number of fourier bins (ws // 2 + 1 by default)
        self.pad = pad
        self.n_mels = n_mels  # number of mel frequency bins
        self.wkwargs = wkwargs
        self.top_db = -80.
        self.f_max = None
        self.f_min = 0.

    def __call__(self, sig):
        transforms = Compose([
            SPECTROGRAM(self.sr, self.ws, self.hop, self.n_fft,
                        self.pad, self.window),
            F2M(self.n_mels, self.sr, self.f_max, self.f_min),
            SPEC2DB("power", self.top_db),
        ])

        spec_mel_db = transforms(sig)

        return spec_mel_db


class MEL(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, tensor):
        if librosa is None:
            print("librosa not installed, cannot create spectrograms")
            return tensor
        L = []
        for i in range(tensor.size(1)):
            nparr = tensor[:, i].numpy()  # (samples, )
            sgram = librosa.feature.melspectrogram(
                nparr, **self.kwargs)  # (n_mels, hops)
            L.append(sgram)
        L = np.stack(L, 2)  # (n_mels, hops, channels)
        tensor = torch.from_numpy(L).type_as(tensor)

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'


class BLC2CBL(object):
    def __call__(self, tensor):
        return tensor.permute(2, 0, 1).contiguous()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MuLawEncoding(object):
    def __init__(self, quantization_channels=256, is_quantized=True, device='cpu'):
        self.qc = quantization_channels
        self.is_quantized = is_quantized
        self.device = device

    def __call__(self, x):
        mu = self.qc - 1.
        if isinstance(x, np.ndarray):
            x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
            x_mu = ((x_mu + 1) / 2 * mu + 0.5)
            if self.is_quantized:
                x_mu = x_mu.astype(int)
            else:
                x_mu /= mu
        elif isinstance(x, (torch.Tensor, torch.LongTensor)):
            if isinstance(x, torch.LongTensor):
                x = x.float()
            mu = torch.tensor([mu], device=self.device)
            x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
            x_mu = ((x_mu + 1) / 2 * mu + 0.5)
            if self.is_quantized:
                x_mu = x_mu.long()
            else:
                x_mu /= mu
        else:
            raise ValueError("Input must be a numpy array or a pytorch tensor!")
        return x_mu

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MuLawExpanding(object):
    def __init__(self, quantization_channels=256, is_quantized=True):
        self.qc = quantization_channels
        self.is_quantized = is_quantized

    def __call__(self, x_mu):
        mu = self.qc - 1.
        if not self.is_quantized:
            x_mu = x_mu * mu
        if isinstance(x_mu, np.ndarray):
            x = ((x_mu) / mu) * 2 - 1.
            x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.) / mu
        elif isinstance(x_mu, (torch.Tensor, torch.LongTensor)):
            if isinstance(x_mu, torch.LongTensor):
                x_mu = x_mu.float()
            mu = torch.FloatTensor([mu])
            x = ((x_mu) / mu) * 2 - 1.
            x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.) / mu
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'
