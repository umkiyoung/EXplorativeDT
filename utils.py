import random
import torch
import numpy as np
import wandb
from pathlib import Path
import uuid

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_np(t):
    """
    convert a torch tensor to a numpy array
    """
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    
    
def wandb_init(variant: dict) -> None:
    run = wandb.init(
        config=variant,
        project=f"{variant['env']}",
        entity="ExDT",
        group=variant["group"],
        name=f"{variant['pretrain_loss_fn']}->{variant['finetune_loss_fn']}-{variant['env']}-{str(uuid.uuid4())[:4]}",
        id=str(uuid.uuid4()),
    )
    wandb.run.save()
    return run