from pytorch_common import *
import librosa


def get_log_mel_spectrogram(filename,
                            n_mels=64,
                            n_fft=1024,
                            hop_length=512,
                            power=2.0):
    wav, sampling_rate = com.file_load(filename)
    mel_spectrogram = librosa.feature.melspectrogram(y=wav,
                                                     sr=sampling_rate,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    return log_mel_spectrogram


def file_to_vector_array_2d(file_name,
                         n_mels=64,
                         steps=20,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a 2d vector array.

    file_name : str
        target .wav file

    return : np.array( np.array( float ) )
        vector array
        * dataset.shape = (dataset_size, n_mels, n_mels)
    """
    # 02 generate melspectrogram using librosa
    y, sr = com.file_load(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vector_array_size = (log_mel_spectrogram.shape[1] - n_mels + 1) // steps

    # 06 generate feature vectors by concatenating multiframes
    vector_array = np.zeros((vector_array_size, n_mels, n_mels))
    for t in range(vector_array_size):
        vector_array[t] = log_mel_spectrogram[:, t*steps:t*steps+n_mels]

    return vector_array


class Task2ImageDataset(torch.utils.data.Dataset):
    """Task 2 dataset to handle samples as 1 channel image.

    Unlike other dataset, set `preprocessed_file` as preprocessed dataset filename.
    For every output, this dataset class crop square image from this original data.

    Number of total samples is `n_sampling` times number of _live data_.
    _Live data_ is all the original data by default, and filtered by splitting functions.

    - For training use, set `random=True`. This will yield randomly cropped square
      image (64x64 for example) from the original sample (64x431 for 10s sample).
    - For validation use, set `random=False`. Image will be cropped from fixed position.

    Augmentation can be flexibly applied to either the output `x` only or `y` only, or both `x` and `y`.
    `aug_x` and `aug_y` control this behavor.

    Data split for training/validation can be done by using:
        `get_index_by_pct()`: generate list of training index.
        `train_split(train_index)`: set live data as original samples listed on `train_index`.
        `val_split(train_index)`: set live data as original samples NOT listed on `train_index`.

    Yields:
        x: square image expected to be used as source.
        y: square image expected to be used as reference for evaluating reconstructed image by training model.
    """

    def __init__(self, preprocessed_file, n_sampling=10, transform=None, augment_tfm=None,
                 normalize=True, random=True, aug_x=True, aug_y=False, debug=False):
        self.n_sampling = n_sampling
        self.transform, self.augment_tfm = transform, augment_tfm
        self.random, self.aug_x, self.aug_y = random, aug_x, aug_y

        self.X = np.load(preprocessed_file)
        if normalize:
            self.X = normalize_0to1(self.X)

        if debug:
            from dlcliche.utils import display
            from dlcliche.math import np_describe
            display(np_describe(self.X[0].cpu().numpy()))

        self.orgX = self.X
  
    def get_index_by_pct(self, split_pct=0.1):
        n = len(self.orgX)
        return random.sample(range(n), k=(n - int(n * split_pct)))

    def train_split(self, train_index):
        self.train_index = train_index
        self.X = self.orgX[train_index]
    
    def val_split(self, train_index):
        n = len(self.orgX)
        self.val_index = [i for i in range(n) if i not in train_index]
        self.X = self.orgX[self.val_index]

    def __len__(self):
        return len(self.X) * self.n_sampling

    def __getitem__(self, index):
        file_index = index // self.n_sampling
        part_index = index % self.n_sampling
        x = self.X[file_index]
        dim, length = x.shape

        # crop square part of sample
        if self.random:
            # random crop
            start = random.randint(0, length - dim)
        else:
            # crop with fixed position
            start = (length // self.n_sampling) * part_index
        start = min(start, length - dim)
        x = x[:, start:start+dim]

        # augmentation transform
        y = x
        if self.augment_tfm is not None:
            tfm_x = self.augment_tfm(x)
            if self.aug_x: x = tfm_x
            if self.aug_y: y = tfm_x

        # transform (convert to tensor here)
        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)
        return x, y
