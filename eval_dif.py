import argparse
import torch

import data_dif as data
from trainer_dif import TrainerMultiple
from utils import *
import pickle
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, )

    parser.add_argument("fingerprint_dir", type=str,
                        help="Directory containing fingerprint and train values")
    parser.add_argument("image_dir", type=str,
                        help="Directory containing real and fake images within 0_real and 1_fake subdirectories")
    parser.add_argument("--epoch", type=int, default=0, required=False,
                        help="Check point epoch to load")
    parser.add_argument("--batch", type=int, default=64, required=False,
                        help="Batch size")

    parsed_args = parser.parse_args()
    return parsed_args


def test_dif_directory(args: argparse.Namespace) -> (float, float):
    '''

    :param args: parser arguments (image directory, fingerprint directory, checkpoint epoch)
    :return: Accuracies for real and fake images
    '''

    model_ep = args.epoch
    images_dir = Path(args.image_dir)
    check_dir = Path(args.fingerprint_dir)

    check_existence(check_dir, False)
    check_existence(images_dir, False)

    with open(check_dir / "train_hypers.pt", 'rb') as pickle_file:
        hyper_pars = pickle.load(pickle_file)

    hyper_pars['Device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyper_pars['Batch Size'] = args.batch

    print(f'Working on {images_dir.stem}')

    real_path_list = [list((images_dir / "0_real").glob('*.' + x)) for x in ['jpg', 'jpeg', 'png']]
    real_path_list = [ele for ele in real_path_list if ele != []][0]

    fake_path_list = [list((images_dir / "1_fake").glob('*.' + x)) for x in ['jpg', 'jpeg', 'png']]
    fake_path_list = [ele for ele in fake_path_list if ele != []][0]

    test_set = data.PRNUData(real_path_list, fake_path_list, hyper_pars, demand_equal=False,
                             train_mode=False)

    trainer = TrainerMultiple(hyper_pars)
    trainer.load_stats(check_dir / f"chk_{model_ep}.pt")

    trainer.test_model(test_set.get_loader())
    acc_f, acc_r = trainer.calc_accuracy(print_res=False)

    return acc_f, acc_r


if __name__ == '__main__':
    acc_f, acc_r = test_dif_directory(parse_arguments())
    print(f'Real Acc. {100 * acc_f:.1f}% | Fake Acc. {100 * acc_r:.1f}% ---> Acc. {50 * (acc_r + acc_f):.1f}%')
