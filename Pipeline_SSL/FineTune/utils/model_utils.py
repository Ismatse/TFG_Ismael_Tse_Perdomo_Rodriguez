import torch
import torch.nn as nn
import os
import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join
matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint["state_dict"])


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def load_train_losses(weights):
    checkpoint = torch.load(weights)
    train_losses = checkpoint["train_losses"]
    return train_losses

def load_val_losses(weights):
    checkpoint = torch.load(weights)
    val_losses = checkpoint["val_losses"]
    return val_losses

def load_mean_dice(weights):
    checkpoint = torch.load(weights)
    mean_dice = checkpoint["mean_dice"]
    return mean_dice

def load_ema_dice(weights):
    checkpoint = torch.load(weights)
    ema_dice = checkpoint["ema_dice"]
    return ema_dice

def plot_progress_png(output_folder, epoca, train_losses, val_losses, mean_dice, ema_dice):
        # we infer the epoch from our internal logging
        epoch = epoca
        sns.set_theme(font_scale=2.5)
        fig, ax = plt.subplots(figsize=(30, 54))
        # regular progress.png as we are used to from previous nnU-Net versions
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(x_values, train_losses[:epoch + 1], color='b', ls='-', label="loss_tr", linewidth=4)
        ax.plot(x_values, val_losses[:epoch + 1], color='r', ls='-', label="loss_val", linewidth=4)
        ax2.plot(x_values, mean_dice[:epoch + 1], color='g', ls='dotted', label="pseudo dice",
                 linewidth=3)
        ax2.plot(x_values, ema_dice[:epoch + 1], color='g', ls='-', label="pseudo dice (mov. avg.)",
                 linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()