from tqdm import tqdm
import torch
import argparse
import data_dif as data
from trainer_dif import TrainerMultiple
from utils import *
from pathlib import Path
import pickle



def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, )

    parser.add_argument("image_dir", type=str,
                        help="Directory containing real and fake images within 0_real and 1_fake subdirectories.")
    parser.add_argument("checkpoint_dir", type=str,
                        help="Directory to save checkpoints. Model is not saved, only fingerprint and statistics.")
    parser.add_argument("--e", type=int, default=100, required=False,
                        help="Amount of train iterations.")
    parser.add_argument("--f", type=int, default=5, required=False,
                        help="Check point frequency.")
    parser.add_argument("--lr", type=float, default=5e-4, required=False,
                        help="Learning rate")
    parser.add_argument("--tr", type=int, default=512, required=False,
                        help="Amount of train samples per real/fake class.")
    parser.add_argument("--cs", type=int, default=256, required=False,
                        help="Crop size (w=h)")
    parser.add_argument("--a", type=float, default=1.0, required=False,
                        help="Alpha - augmentations")
    parser.add_argument("--b", type=bool, default=False, required=False,
                        help="Booster loss")
    parser.add_argument("--bs", type=int, default=64, required=False,
                        help="Booster loss")

    parsed_args = parser.parse_args()
    return parsed_args


def train_model(args: argparse.Namespace) -> None:

    data_root = Path(args.image_dir)
    check_dir = Path(args.checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hyper_pars = {'Epochs': args.e, 'Factor': args.f, 'Noise Type': 'uniform', "Train Size": args.tr,
                  'Noise STD': 0.03, 'Inp. Channel': 16, 'Batch Size': 64,
                  'LR': 5e-4, 'Device': device, 'Crop Size': (args.cs, args.cs), 'Margin': 0.01,
                  'Out. Channel': 3, 'Arch.': 32, 'Depth': 4, 'Alpha': args.a, 'Boost': args.b,
                  'Concat': [1, 1, 1, 1]}

    check_existence(check_dir, True)
    check_existence(data_root, False)

    print('Preparing Data Sets...')

    real_data_root = data_root / "0_real"
    fake_data_root = data_root / "1_fake"

    train_set, _, file_split = data.prep_data_sets(real_data_root, fake_data_root, hyper_pars)
    train_loader = train_set.get_loader()

    pickle.dump(file_split, open((check_dir / 'file_split.pt'), 'wb'))
    pickle.dump(hyper_pars, open((check_dir / 'train_hypers.pt'), 'wb'))

    print('Preparing Trainer...')
    trainer = TrainerMultiple(hyper_pars).to(hyper_pars['Device'])

    epochs = list(range(1, hyper_pars['Epochs'] + 1))
    pbar = tqdm(total=len(epochs), desc='')

    for ep in epochs:
        pbar.update()

        for residual, labels in train_loader:
            trainer.train_step(residual, labels)

        if (ep % hyper_pars['Factor']) == 0:
            if ep > 0:
                trainer.save_stats(check_dir / ('chk_' + str(ep) + '.pt'))

        pbar.postfix = f'Loss C {np.mean(trainer.train_loss[-10:]):.3f} ' + \
                       f'| Fake C {np.mean(trainer.train_corr_f[-10:]):.3f} | Real C {np.mean(trainer.train_corr_r[-10:]):.3f}'

    trainer.save_stats(check_dir / ('chk_' + str(hyper_pars['Epochs']) + '.pt'))
    torch.cuda.empty_cache()


if __name__ == '__main__':
    train_model(parse_arguments())
