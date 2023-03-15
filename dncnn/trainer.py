
from torch import optim, nn
import torch
from dncnn.model import DnCNN


relu = nn.ReLU()

class TrainerDnCNN(nn.Module):
    def __init__(self, hyperparams):
        super(TrainerDnCNN, self).__init__()

        # Hyperparameters
        self.device = hyperparams['Device']
        self.init_lr = hyperparams['LR']
        self.ch_o = hyperparams['Out. Channel']
        self.m = hyperparams['Margin']
        self.batch_size = hyperparams['Batch Size']
        self.crop_size = hyperparams['Crop Size']
        self.depth = hyperparams['Depth']
        self.crop_b = hyperparams['Crop Batch']

        self.train_loss = []

        self.train_mean_r = []
        self.train_mean_f = []

        self.test_mean_r = []
        self.test_mean_f = []

        # Model initialization
        self.denoiser = DnCNN(self.ch_o, self.depth).to(self.device)
        self.optimizer = optim.AdamW(self.denoiser.parameters(), lr=self.init_lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.998)

        self.loss_fun = nn.MSELoss(reduction='sum')
        self.loss_bce = nn.BCELoss(reduction='none')


def load_model(trainer, path, device):
    if device.type == 'cpu':
        data_dict = torch.load(path, map_location=torch.device('cpu'))
    else:
        data_dict = torch.load(path)

    try:
        trainer.unet.load_state_dict(data_dict['G state'])
        trainer.train_loss = data_dict['Train G Loss']

        return len(trainer.train_loss)
    except:
        return data_dict

