import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

prnu_tens = transforms.ToTensor()

def shuffle_split(data_list, n_train):
    idx_list = list(range(len(data_list)))
    np.random.shuffle(idx_list)
    train_idx_list = idx_list[:n_train]
    test_idx_list = idx_list[n_train:]

    return np.array(data_list)[train_idx_list], np.array(data_list)[test_idx_list]


def prep_data_sets(real_dir, fake_dir, h_dict, test_only=False):
    real_path_list = [list(real_dir.glob('*.' + x)) for x in ['jpg', 'jpeg', 'png']]
    real_paths = [ele for ele in real_path_list if ele != []][0]

    fake_path_list = [list(fake_dir.glob('*.' + x)) for x in ['jpg', 'jpeg', 'png']]
    fake_paths = [ele for ele in fake_path_list if ele != []][0]

    if not test_only:
        n_train = h_dict["Train Size"]

        n_real = len(real_paths)
        n_fake = len(fake_paths)
        n = n_real

        if n_real > n_fake:

            idx_list = list(range(n_real))
            np.random.shuffle(idx_list)
            img_idx_list = idx_list[:n_fake]
            real_paths = [real_paths[i] for i in img_idx_list]
            n = n_fake

        elif n_real < n_fake:

            idx_list = list(range(n_fake))
            np.random.shuffle(idx_list)
            img_idx_list = idx_list[:n_real]
            fake_paths = [fake_paths[i] for i in img_idx_list]
            n = n_real

        if n < n_train:
            raise Exception(f"{n_train} images were requested for train, but there are only {n} images.")

        train_real_paths, test_real_paths = shuffle_split(real_paths, n_train)
        train_fake_paths, test_fake_paths = shuffle_split(fake_paths, n_train)

        train_set = PRNUData(train_real_paths, train_fake_paths, h_dict)
        test_set = PRNUData(test_real_paths, test_fake_paths, h_dict)

        file_dict = {"Train Real": train_real_paths,
                     "Test Real": test_real_paths,
                     "Train Fake": train_fake_paths,
                     "Test Fake": test_fake_paths}

        return train_set, test_set, file_dict

    else:
        test_set = PRNUData(real_path_list, fake_path_list, h_dict)

        file_dict = {"Real": real_path_list,
                     "Fake": fake_path_list}

        return None, test_set, file_dict


def rescale_img(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def load_pil_image(img_path, img_size=None):
    img = Image.open(img_path).convert('RGB')

    if img_size is not None:
        w, h = img.size
        left = (w - img_size[1]) / 2
        top = (h - img_size[0]) / 2
        right = (w + img_size[1]) / 2
        bottom = (h + img_size[0]) / 2

        img = img.crop((left, top, right, bottom))

    return img

def produce_fft(finger_npy):
    fft_f = np.fft.fft2(finger_npy - finger_npy.mean(), axes=(0, 1), norm='forward')

    finger_spec = rescale_img(np.log(np.abs(fft_f)))
    finger_spec = np.fft.fftshift(finger_spec) ** 4

    return finger_spec

class PRNUData(Dataset):
    def __init__(self, real_paths, fake_paths, hyper_pars,
                 demand_equal=True,
                 train_mode=True):

        self.real_paths = real_paths
        self.fake_paths = fake_paths
        self.file_list = None

        self.real_labels = None
        self.fake_labels = None
        self.label_list = None

        self.crop_size = hyper_pars['Crop Size']
        self.batch_size = hyper_pars['Batch Size']

        self.prep_inputs(demand_equal)
        self.init_loader()
        self.train_mode = train_mode

    def prep_inputs(self, demand_equal):

        n_real = len(self.real_paths)
        n_fake = len(self.fake_paths)
        n = n_real

        if demand_equal:
            if n_real > n_fake:
                self.real_paths = self.real_paths[:n_fake]
                n = n_fake

            elif n_real < n_fake:
                self.fake_paths = self.fake_paths[:n_real]
                n = n_real

        self.real_labels = torch.zeros((len(self.real_paths),))
        self.fake_labels = torch.ones((len(self.fake_paths),))

        self.file_list = np.array(list(self.real_paths) + list(self.fake_paths))
        self.label_list = torch.cat((self.real_labels, self.fake_labels), dim=0).type(torch.bool)

    def init_loader(self):
        self.loader = DataLoader(self, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def get_loader(self):
        return self.loader

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]

        image = load_pil_image(img_path, self.crop_size)
        label = self.label_list[idx]
        image = np.array(image)

        image = torch.tensor(image.transpose((2, 0, 1))).type(torch.float32).div(255)

        return image, label
