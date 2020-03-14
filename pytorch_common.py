import torch
from torch import nn
import torch.nn.functional as F
import torchsummary
import torch
import pytorch_lightning as pl
import random
import common as com
from dlcliche.utils import *


class _LinearUnit(torch.nn.Module):
    """For use in Task2Baseline model."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)
        self.bn = torch.nn.BatchNorm1d(out_dim)

    def forward(self, x):
        return torch.relu(self.bn(self.lin(x.view(x.size(0), -1))))


class Task2Baseline(torch.nn.Module):
    """PyTorch version of the baseline model."""
    def __init__(self):
        super().__init__()
        self.unit1 = _LinearUnit(640, 128)
        self.unit2 = _LinearUnit(128, 128)
        self.unit3 = _LinearUnit(128, 128)
        self.unit4 = _LinearUnit(128, 128)
        self.unit5 = _LinearUnit(128, 8)
        self.unit6 = _LinearUnit(8, 128)
        self.unit7 = _LinearUnit(128, 128)
        self.unit8 = _LinearUnit(128, 128)
        self.unit9 = _LinearUnit(128, 128)
        self.output = torch.nn.Linear(128, 640)

    def forward(self, x):
        shape = x.shape
        x = self.unit1(x.view(x.size(0), -1))
        x = self.unit2(x)
        x = self.unit3(x)
        x = self.unit4(x)
        x = self.unit5(x)
        x = self.unit6(x)
        x = self.unit7(x)
        x = self.unit8(x)
        x = self.unit9(x)
        return self.output(x).view(shape)


def load_model(model_file, mode='baseline', summary=True, **kwargs):
    """Load task2 models, and show summary."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = (Task2Baseline() if mode == 'baseline'
             else VAE(device, x_dim=640, **kwargs) if mode == 'vae'
             else 'Unknown mode')
    if model_file is not None:
        model.load_state_dict(torch.load(model_file))
    if summary:
        torchsummary.summary(model.to(device), input_size=(1, 640))
    return model


class VAE(nn.Module):
    """Mostly borrowed from PyTorch example.
    Thanks to https://github.com/pytorch/examples/blob/master/vae/main.py
    """

    def __init__(self, device, x_dim, h_dim=400, z_dim=20):
        super().__init__()
        self.x_dim = x_dim

        self.fc11 = nn.Linear(x_dim, h_dim)
        self.fc12 = nn.Linear(h_dim, h_dim)
        self.fc13 = nn.Linear(h_dim, h_dim)
        self.fc14 = nn.Linear(h_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.fc31 = nn.Linear(z_dim, h_dim)
        self.fc32 = nn.Linear(h_dim, h_dim)
        self.fc33 = nn.Linear(h_dim, h_dim)
        self.fc34 = nn.Linear(h_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, x_dim)

    def encode(self, x):
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        #x = F.relu(self.fc13(x))
        #x = F.relu(self.fc14(x))
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = F.relu(self.fc31(z))
        h = F.relu(self.fc32(h))
        #h = F.relu(self.fc33(h))
        #h = F.relu(self.fc34(h))
        return torch.sigmoid(self.fc4(h))

    def forward_all(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar

    def forward(self, x):
        yhat, _, _, _ = self.forward_all(x)
        return yhat


def VAE_loss_function(recon_x, x, mu, logvar, reconst_loss='mse', a_RECONST=1., a_KLD=1., x_dim=640):
    """Reconstruction + KL divergence losses summed over all elements and batch.
    Thanks to https://github.com/pytorch/examples/blob/master/vae/main.py"""

    func = (F.mse_loss if reconst_loss == 'mse'
            else F.binary_cross_entropy if reconst_loss == 'bce'
            else 'Unknown reconst_loss')
    RECONST = func(recon_x, x.view(-1, x_dim), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return RECONST*a_RECONST + KLD*a_KLD


class Task2Dataset(torch.utils.data.Dataset):
    """PyTorch dataset class for task2. Caching to a file supported.

    Args:
        n_mels, frames, n_fft, hop_length, power, transform: Audio conversion settings.
        normalize: Normalize data value range from [-90, 24] to [0, 1] for VAE, False by default.
        cache_to: Cache filename or None by default, use this for your iterative development.
    """

    def __init__(self, files, n_mels, frames, n_fft, hop_length, power, transform,
                 normalize=False, cache_to=None, debug=False):
        self.transform = transform
        self.files = files
        self.n_mels, self.frames, self.n_fft = n_mels, frames, n_fft
        self.hop_length, self.power = hop_length, power
        # load cache or convert all the data for the first time
        if cache_to is not None and Path(cache_to).exists():
            com.logger.info(f'Loading cached {Path(cache_to).name}')
            self.X = np.load(cache_to)
        else:
            self.X = com.list_to_vector_array(self.files,
                             n_mels=self.n_mels,
                             frames=self.frames,
                             n_fft=self.n_fft,
                             hop_length=self.hop_length,
                             power=self.power)
            if cache_to is not None:
                np.save(cache_to, self.X)

        if normalize:
            # Normalize to range from [-90, 24] to [0, 1] based on dataset quick stat check.
            self.X = (self.X + 90.) / (24. + 90.)
            self.X = np.clip(self.X, 0., 1.)
        if debug:
            from dlcliche.utils import display
            from dlcliche.math import np_describe
            for x, _ in dl:
                display(np_describe(x.cpu().numpy()))
                break  # Show first range only

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        x = self.transform(x)
        return x, x


class Task2Lightning(pl.LightningModule):
    """Task2 PyTorch Lightning class, for training only."""

    def __init__(self, device, params, files, mode='baseline', normalize=False, **model_kwargs):
        super().__init__()
        self.device = device
        self.params = params
        self.normalize = normalize
        self.model = load_model(None, mode=mode, **model_kwargs)
        self.mseloss = torch.nn.MSELoss()
        # split data files
        n_val = int(params.fit.validation_split * len(files))
        self.val_files = random.sample(files, n_val)
        self.train_files = [f for f in files if f not in self.val_files]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.mseloss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': self.mseloss(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params.fit.lr,
                                betas=(self.params.fit.b1, self.params.fit.b2),
                                weight_decay=self.params.fit.weight_decay)

    def _get_dl(self, for_what):
        files = self.train_files if for_what == 'train' else self.val_files
        cache_file = f'{self.params.model_directory}/__cache_{str(files[0]).split("/")[-3]}_{for_what}.npy'
        ds = Task2Dataset(files,
                          n_mels=self.params.feature.n_mels,
                          frames=self.params.feature.frames,
                          n_fft=self.params.feature.n_fft,
                          hop_length=self.params.feature.hop_length,
                          power=self.params.feature.power,
                          transform=com.ToTensor1d(device=self.device),
                          normalize=self.normalize,
                          cache_to=cache_file)
        return torch.utils.data.DataLoader(ds, batch_size=self.params.fit.batch_size,
                          shuffle=(self.params.fit.shuffle if for_what == 'train' else False))

    def train_dataloader(self):
        return self._get_dl('train')

    def val_dataloader(self):
        return self._get_dl('val')
