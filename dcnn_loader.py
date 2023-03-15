from pathlib import Path
import torch
import numpy as np
from dncnn.trainer import TrainerDnCNN, load_model


dncnn_root = Path("dncnn")

def load_denoiser(device: str, trainable:bool=False)-> torch.nn.Module:

    denoiser_prnu_np = np.load(str(dncnn_root / r"clean_real.npy"), allow_pickle=True)

    trainer = load_model(TrainerDnCNN, dncnn_root / f"chk_2000.pt", device)
    model = trainer.denoiser.to(device)

    denoiser_prnu = torch.tensor(denoiser_prnu_np.transpose((2, 0, 1))).to(device).unsqueeze(0)

    model.prnu = denoiser_prnu

    if not trainable:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    return model
