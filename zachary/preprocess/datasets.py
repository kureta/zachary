from zachary.preprocess.base import BaseDataset, Configuration


class AtemporalDataset(BaseDataset):
    def __init__(self, conf=Configuration()):
        super(AtemporalDataset, self).__init__(conf=conf)

    def __len__(self):
        return self.spectra.shape[0]

    def __getitem__(self, index):
        return self.spectra[index]


class TemporalDataset(BaseDataset):
    def __init__(self, conf=Configuration(), example_length=64, example_hop_length=32):
        super(TemporalDataset, self).__init__(conf=conf)

        self._example_length = example_length
        self._example_hop_length = example_hop_length
        self.strided_examples = None
        self.update_strided()

    def update_strided(self):
        stride = self.spectra.stride()
        shape = self.spectra.shape
        n_examples = (shape[0] - self.example_length) // self.example_hop_length + 1
        self.strided_examples = self.spectra.as_strided(
            (n_examples, shape[1], self.example_length),
            (stride[0] * self.example_hop_length, stride[1], stride[0]))

    @property
    def example_length(self):
        return self._example_length

    @example_length.setter
    def example_length(self, value):
        self._example_length = value
        self.update_strided()

    @property
    def example_hop_length(self):
        return self._example_hop_length

    @example_hop_length.setter
    def example_hop_length(self, value):
        self._example_hop_length = value
        self.update_strided()

    def __len__(self):
        return self.strided_examples.shape[0]

    def __getitem__(self, index):
        return self.strided_examples[index]
