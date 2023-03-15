import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import data_dif as data
from trainer_dif import TrainerSingle
from utils import *


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--size", type=tuple, default=(128, 128), required=False,
                        help="Crop size in pixels (h,w)")
    parser.add_argument("--output_dir", type=str, default='out', required=False,
                        help="Path to the directory for saving images")
    parser.add_argument("--epochs", type=int, default=1000, required=False,
                        help="Amount of epoch for train")

    parsed_args = parser.parse_args()
    return parsed_args


def preform_blank(args: argparse.Namespace) -> None:
    epochs = args.epochs
    output_dir = Path(args.output_dir)
    crop_size = args.size

    check_existence(output_dir, True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hyper_pars = {'Epochs': epochs, 'Factor': 100, 'Noise Type': 'uniform',
                  'Noise STD': 0.03, 'Inp. Channel': 16,
                  'LR': 5e-5, 'Device': device, 'Crop Size': crop_size, 'Margin': 1.,
                  'Out. Channel': 3, 'Arch.': 32, 'Depth': 4,
                  'Concat': [1, 1, 1, 1]}

    # Generate tensor of the gray images
    img_lp = 0.5 * torch.ones((1, 3, *hyper_pars['Crop Size']))
    img_lp = img_lp.to(device).float()

    # Initialize trainer
    trainer = TrainerSingle(hyper_pars).to(hyper_pars['Device'])
    train_step_fun = trainer.train_step_blank

    # Initialize progress bar
    epochs = list(range(1, hyper_pars['Epochs'] + 1))
    pbar = tqdm(total=len(epochs), desc='')
    ep = 0

    fin_list = {}

    for ep in epochs:
        pbar.update()
        train_step_fun(img_lp)

        if (ep % hyper_pars['Factor']) == 0:
            if ep > 0:
                fin_np = trainer.produce_fingerprint(True)
                fin_list[ep] = fin_np

        pbar.postfix = f'Loss {trainer.train_loss[- 1]:.5f}'

    fin_np = trainer.produce_fingerprint(True)
    fin_list[ep] = fin_np
    fin_fft = data.show_prnu_fft(fin_list[ep], r'$\hat{Y}$' + f' {ep}')

    # Saving output history and images from the last epoch
    np.save(str(output_dir / 'fin_history.npy'), fin_list)
    Image.fromarray((255*fin_np).astype('uint8')).save(output_dir / 'fin_image.png')
    Image.fromarray((255*data.rescale_img(fin_fft)).astype('uint8')).save(output_dir / 'fin_fft.png')


if __name__ == '__main__':
    preform_blank(parse_arguments())
