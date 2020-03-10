import torch
import torchsummary


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


def load_model(model_file, summary=True):
    """Load Task2Baseline model, and show summary."""
    model = Task2Baseline()
    model.load_state_dict(torch.load(model_file))
    if summary:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torchsummary.summary(model.to(device), input_size=(1, 640))
    return model

