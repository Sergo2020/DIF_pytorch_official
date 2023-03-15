import matplotlib.pyplot as plt
import numpy as np

from torch import optim, nn
import torch
from tqdm import tqdm

from dcnn_loader import load_denoiser
import model
from utils import calc_even_size, produce_spectrum

relu = nn.ReLU()


class TrainerMultiple(nn.Module):
    def __init__(self, hyperparams):
        super(TrainerMultiple, self).__init__()

        # Hyperparameters
        self.device = hyperparams['Device']
        self.init_lr = hyperparams['LR']
        self.ch_i = hyperparams['Inp. Channel']
        self.ch_o = hyperparams['Out. Channel']
        self.arch = hyperparams['Arch.']
        self.depth = hyperparams['Depth']
        self.concat = np.array(hyperparams['Concat'])
        self.m = hyperparams['Margin']
        self.batch_size = hyperparams['Batch Size']
        self.alpha = hyperparams['Alpha']
        try:
            self.boost = hyperparams['Boost']
        except:
            self.boost = False

        self.train_loss = []
        self.train_corr_r = None
        self.train_corr_f = None

        self.test_loss = []
        self.test_corr_r = []
        self.test_corr_f = []
        self.test_labels = []

        self.noise_type = hyperparams['Noise Type']
        self.noise_std = hyperparams['Noise STD']
        self.noise_channel = hyperparams['Inp. Channel']
        self.crop_size = hyperparams['Crop Size']

        d_h, n_h, d_w, n_w = calc_even_size(self.crop_size, self.depth)
        self.crop_size = (n_h - d_h, n_w - d_w)
        self.d_h, self.n_h = d_h, n_h
        self.d_w, self.n_w = d_w, n_w

        # Model initialization
        self.noise = None

        self.denoiser = load_denoiser(self.device)
        self.unet = model.Unet(self.device, self.ch_i, self.ch_o, self.arch,
                               activ='leak', depth=self.depth, concat=self.concat).to(self.device)
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=self.init_lr)

        self.loss_mse = nn.MSELoss()

        self.init_train()

    def norm_val(self, arr):
        return (arr - arr.mean((1, 2, 3)).view(-1, 1, 1, 1)) / (arr.std((1, 2, 3)).view(-1, 1, 1, 1) + 1e-8)

    def init_train(self, n=1):
        self.noise = init_dummy(n, self.noise_type, self.crop_size, self.noise_channel)
        self.fingerprint = None

    def prep_noise(self, var=-1):
        if var == -1:
            return self.noise + torch.randn_like(self.noise.detach()) * self.noise_std
        else:
            return self.noise + torch.randn_like(self.noise.detach()) * var

    def corr_fun(self, out, target):
        # Pearson Correlation Coefficient (NNC(0,0))
        out = self.norm_val(out)
        target = self.norm_val(target)

        return out * target

    def loss_contrast(self, corrs, labs):
        # Label: 0 - Real, 1 - Fake
        # Similarity: 0 - Similar, 1 - Different
        n = len(corrs) // 2
        corr_a = corrs[:n]
        lab_a = labs[:n]

        corr_b = corrs[n:]
        lab_b = labs[n:]

        sim_label = torch.bitwise_xor(lab_a, lab_b).type(torch.float64)  # .view(-1, 1)
        corr_delta = torch.sqrt(((corr_a - corr_b) ** 2))
        loss = sim_label * (self.m - corr_delta) + (1. - sim_label) * corr_delta

        return relu(loss)

    def train_step(self, images, labels):

        images = images.to(self.device)
        labels = labels.to(self.device)

        self.unet.train()
        self.optimizer.zero_grad()

        residuals = self.denoiser.denoise(images).detach()
        alpha = (1 - self.alpha) * torch.rand((len(images), 1, 1, 1)).to(self.device) + self.alpha
        residuals = alpha * residuals

        f_mean = residuals[labels].mean(0, keepdims=True)
        r_mean = residuals[~labels].mean(0, keepdims=True)

        residuals = torch.cat((residuals, f_mean, r_mean), dim=0)

        dmy = self.prep_noise().to(self.device)
        out = self.unet(dmy).repeat(len(images) + 2, 1, 1, 1)

        corr = self.corr_fun(out, residuals)

        loss = self.loss_contrast(corr[:-2].mean((1, 2, 3)), labels).mean() / self.m

        if self.boost:
            corr_mean_d = torch.sqrt((corr[-2].mean() - corr[-1].mean()) ** 2)
            loss_b = relu(self.m - corr_mean_d) / self.m
            loss += loss_b
            loss *= 0.5

        loss.backward()
        self.optimizer.step()

        if self.fingerprint is None:
            self.fingerprint = out[0:1].detach()
        else:
            self.fingerprint = self.fingerprint * 0.99 + out[0:1].detach() * (1 - 0.99)

        corr = self.corr_fun(self.fingerprint.repeat(len(images), 1, 1, 1), residuals[:-2]).mean((1, 2, 3))

        self.train_loss.append(loss.item())

        if self.train_corr_r is None:
            self.train_corr_r = [corr[~labels].mean().item()]
            self.train_corr_f = [corr[labels].mean().item()]
        else:
            corr_r = corr[~labels]
            corr_f = corr[labels]
            self.train_corr_r.append(corr_r.mean().item())
            self.train_corr_f.append(corr_f.mean().item())

    def reset_test(self):
        self.test_corr_r = None
        self.test_corr_f = None

        self.test_loss = []
        self.test_labels = []

    def test_model(self, test_loader, custom_finger=None):

        self.reset_test()
        self.calc_centers()

        if custom_finger is None:
            fingerprint = self.fingerprint.to(self.device)
            fingerprint.repeat((self.batch_size, 1, 1, 1))


        else:
            if isinstance(custom_finger, np.ndarray):
                custom_finger = torch.Tensor(custom_finger.transpose((2, 0, 1))).type(torch.float32)

            fingerprint = custom_finger.to(self.device)
            fingerprint = fingerprint.repeat((self.batch_size, 1, 1, 1))

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing Model'):
                images = images.to(self.device)
                labels = labels.to(self.device)

                residuals = self.denoiser.denoise(images).float()

                corr = self.corr_fun(fingerprint, residuals)
                loss = self.loss_contrast(corr.mean((1, 2, 3)), labels) / self.m

                corr = corr.mean((1, 2, 3))

                self.test_loss = self.test_loss + loss.tolist()
                self.test_labels = self.test_labels + labels.tolist()

                if self.test_corr_r is None:
                    self.test_corr_r = corr[~labels].cpu().numpy()
                    self.test_corr_f = corr[labels].cpu().numpy()
                else:
                    self.test_corr_r = np.append(self.test_corr_r, corr[~labels].cpu().numpy(), axis=0)
                    self.test_corr_f = np.append(self.test_corr_f, corr[labels].cpu().numpy(), axis=0)

    def produce_fingerprint(self, np=True):
        with torch.no_grad():
            out = self.fingerprint[0]

        if np:
            return out.cpu().numpy().transpose((1, 2, 0))
        else:
            return out

    def plot_loss(self, train=True):
        plt.figure(figsize=(10, 6))

        if train:
            plt.scatter(np.arange(1, len(self.train_loss) + 1), self.train_loss, s=3, label='Loss', c='g')
            plt.xlabel('Batch Index')
            plt.ylabel('Mean Sample Loss')
            plt.title('Train Loss')

        else:
            self.test_labels = np.array(self.test_labels)
            colors = np.array([(1., 0., 0.)] * len(self.test_labels))
            colors[self.test_labels == 0] = (0., 1., 0.)

            plt.scatter(np.arange(1, len(self.test_loss) + 1), self.test_loss, s=3, label='Loss', c=colors)
            plt.xlabel('Label Index')
            plt.ylabel('Sample Loss')
            plt.title('Test Loss')

        plt.grid(True)
        plt.ylim([0., 1.0])
        plt.legend(fontsize=12)
        plt.tight_layout()

        plt.show()

    def show_fingerprint(self):
        finger = self.produce_fingerprint()
        finger = 0.5 * finger + 0.5

        plt.figure(figsize=(4, 4))

        plt.imshow(finger)
        plt.axis(False)
        plt.title('Fingerprint')

        plt.show()

        dct_finger = produce_spectrum(finger)
        dct_finger = (dct_finger - dct_finger.min()) / (dct_finger.max() - dct_finger.min())

        plt.figure(figsize=(4, 4))

        plt.imshow(dct_finger, 'bone')
        plt.axis(False)
        plt.title('Fingerprint FFT')

        plt.show()

    def plot_corr(self, train=True):

        plt.figure(figsize=(10, 6))

        if train:

            plt.scatter(np.arange(len(self.train_corr_r)), self.train_corr_r, s=3,
                        label='Real Corr.', c='g')
            plt.scatter(np.arange(len(self.train_corr_f)), self.train_corr_f, s=3,
                        label='Fake Corr.', c='r')

            plt.xlabel('Batch Index')
            plt.ylabel('Mean Sample Corr.')
            plt.title('Train Correlation')

        else:

            plt.scatter(np.arange(1, len(self.test_corr_r) + 1), self.test_corr_r, s=3, label='Real Corr.', c='g')
            plt.scatter(np.arange(1, len(self.test_corr_f) + 1), self.test_corr_f, s=3, label='Fake Corr.', c='r')
            plt.xlabel('Label Index')
            plt.title('Test Correlation')
            plt.ylabel('Sample Corr.')

        plt.grid(True)
        plt.legend(fontsize=12)

        plt.show()

    def calc_centers(self):
        self.mu_real = np.mean(self.train_corr_r[-20:])
        self.mu_fake = np.mean(self.train_corr_f[-20:])

    def calc_distance(self):
        dist_real = distance(self.test_corr_r, self.mu_real, self.mu_fake)
        dist_fake = distance(self.test_corr_f, self.mu_real, self.mu_fake)

        return dist_fake, dist_real

    def calc_accuracy(self, val=None, print_res=True):

        if val is not None:
            dist = distance(val, self.mu_real, self.mu_fake)
            cls = np.argmin(dist, axis=1)

            return dist[0], cls[0]

        else:
            # Real - 0, Fake - 1
            dist_real = distance(self.test_corr_r, self.mu_real, self.mu_fake)
            dist_fake = distance(self.test_corr_f, self.mu_real, self.mu_fake)

            class_real = np.argmin(dist_real, axis=1) == 0
            class_fake = np.argmin(dist_fake, axis=1) == 1

            acc_real = class_real.sum() / len(class_real)
            acc_fake = class_fake.sum() / len(class_fake)

            if print_res:
                print("Accuracy by cluster means:")
                print(f" Real samples: {acc_real:.2f}")
                print(f" Fake samples: {acc_fake:.2f}")
                print(f" All samples: {(acc_real + acc_fake) / 2.:.2f}")

            return acc_fake, acc_real

    def show_prnu_density(self, title=None):

        corr_r = self.test_corr_r
        corr_f = self.test_corr_f

        fig, ax = plt.subplots(figsize=(4, 4))
        for val, data_type, mu in zip([corr_r, corr_f], ["Real", "Fake"], [self.mu_real, self.mu_fake]):
            hist = np.histogram(val, bins=100)
            width = (hist[1][-1] - hist[1][0]) / 100
            ax.bar(hist[1][1:], hist[0], width, alpha=0.5, label=f'{data_type}')

            ax.axvline(x=mu, ymin=0, ymax=np.max(hist[0]), linestyle="--", color='k',
                       label=r'$\mu_{' + f'{data_type}' + '}$')

        ax.set_ylabel('Count')
        ax.set_xlabel(r'$\rho$')
        ax.legend()
        ax.grid()
        if title is not None:
            plt.title(title)

        fig.tight_layout()
        fig.show()

    def save_stats(self, path):
        self.calc_centers()

        data_dict = {'Fingerprint': self.fingerprint,
                     'Train Real': self.train_corr_r,
                     'Train Fake': self.train_corr_f,
                     'Loss': self.train_loss}

        torch.save(data_dict, path)

    def load_stats(self, path):
        if self.device.type == 'cpu':
            data_dict = torch.load(path, map_location=torch.device('cpu'))
        else:
            data_dict = torch.load(path)

        self.train_loss = data_dict['Loss']
        self.train_corr_r = data_dict['Train Real']
        self.train_corr_f = data_dict['Train Fake']
        self.fingerprint = data_dict['Fingerprint']


class TrainerSingle(nn.Module):
    def __init__(self, hyperparams):
        super(TrainerSingle, self).__init__()

        # Hyperparameters
        self.device = hyperparams['Device']
        self.init_lr = hyperparams['LR']
        self.ch_i = hyperparams['Inp. Channel']
        self.ch_o = hyperparams['Out. Channel']
        self.arch = hyperparams['Arch.']
        self.depth = hyperparams['Depth']
        self.concat = np.array(hyperparams['Concat'])
        self.crop_size = hyperparams['Crop Size']

        self.train_corr_hp = []
        self.train_corr_lp = []
        self.train_loss = []

        self.noise_type = hyperparams['Noise Type']
        self.noise_std = hyperparams['Noise STD']
        self.noise_channel = hyperparams['Inp. Channel']
        self.noise = None

        d_h, n_h, d_w, n_w = calc_even_size(self.crop_size, self.depth)
        self.crop_size = (n_h - d_h, n_w - d_w)
        self.d_h, self.n_h = d_h, n_h
        self.d_w, self.n_w = d_w, n_w

        self.init_train()

        # Model initialization
        self.AE = model.Unet(self.device, self.ch_i, self.ch_o, self.arch,
                             activ='leak', depth=self.depth, concat=self.concat).to(self.device)
        self.optimizer = optim.AdamW(self.AE.parameters(), lr=self.init_lr)

        self.loss_mse = nn.MSELoss()

    def init_train(self):  # Check what STD is better for initial noise
        self.noise = init_dummy(1, self.noise_type, self.crop_size, self.noise_channel)

    def prep_noise(self, var=-1):
        if var == -1:
            return self.noise + torch.randn_like(self.noise.detach()) * self.noise_std
        else:
            return self.noise + torch.randn_like(self.noise.detach()) * var

    def train_step_blank(self, blank):
        self.AE.train()

        self.optimizer.zero_grad()

        dmy = self.prep_noise().to(self.device)

        out = self.AE(dmy)
        loss = self.loss_mse(out, blank).mean()

        loss.backward()
        self.optimizer.step()
        self.train_loss.append(loss.item())

        return out.detach().cpu()

    def plot_loss_corr(self):
        plt.figure(figsize=(10, 2 * 6))

        plt.subplot(2, 1, 1)
        plt.scatter(np.arange(1, len(self.train_loss) + 1), self.train_loss, c='g')
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.subplot(2, 1, 2)
        plt.scatter(np.arange(1, len(self.train_corr_hp) + 1), self.train_corr_hp, c='r', label='HP')
        plt.scatter(np.arange(1, len(self.train_corr_lp) + 1), self.train_corr_lp, c='g', label='LP')
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('Correlation')

        plt.legend()
        plt.show()

    def produce_fingerprint(self, np=True):
        self.AE.eval()
        with torch.no_grad():
            out = self.AE(self.prep_noise().to(self.device)[0:1])[0]
            out = 0.5 * out + .5
        if np:
            return out.cpu().numpy().transpose((1, 2, 0))
        else:
            return out


def distance(arr, mu_a, mu_b):
    dist_arr2a = np.sqrt(((arr - mu_a) ** 2)).reshape((-1, 1))
    dist_arr2b = np.sqrt(((arr - mu_b) ** 2)).reshape((-1, 1))
    return np.concatenate((dist_arr2a, dist_arr2b), axis=1)


def init_dummy(bs, noise_type, img_dims, ch_n, var=0.1):
    if noise_type == 'uniform':
        img = var * torch.rand((bs, ch_n, img_dims[0], img_dims[1]))
    elif noise_type == 'normal':
        img = var * torch.randn((bs, ch_n, img_dims[0], img_dims[1]))
    elif noise_type == 'mesh':
        assert ch_n == 2
        X, Y = np.meshgrid(np.arange(0, img_dims[1]) / float(img_dims[1] - 1),
                           np.arange(0, img_dims[0]) / float(img_dims[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        img = torch.tensor(meshgrid).unsqueeze(0).type(torch.float)

    elif noise_type == 'special':
        X, Y = np.meshgrid(np.arange(0, img_dims[1]) / float(img_dims[1] - 1),
                           np.arange(0, img_dims[0]) / float(img_dims[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        img = torch.tensor(meshgrid).unsqueeze(0).type(torch.float)
        img = torch.cat((img, torch.ones((1, 1, img_dims[0], img_dims[1]))), dim=1)
    return img
