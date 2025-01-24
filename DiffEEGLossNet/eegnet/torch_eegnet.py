######################################################Plot confusion matrix####################################################
import torch
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score
import matplotlib.colors as mcolors
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ELU, AvgPool2d, Dropout, Flatten, Linear, CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR

class EEGNet(Module):
    def __init__(self, sampling_rate: int, N: int, L: int, C: int,
                 F1=8, D=2, F2=16, dropout_rate=0.5):
        """
            Args:
                sampling_rate: Sampling rate of data
                N: Number of classes
                L: Signal length
                C: Number of channels
                F1: Number of temporal filters
                D: Depth multiplier
                F2: Number of pointwise filters
                dropout_rate: Dropout rate
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.N = N
        self.L = L
        self.C = C
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropout_rate = dropout_rate

        self.block1 = Sequential(OrderedDict([
            ('conv', Conv2d(1, F1, (1, sampling_rate//2), padding='same', bias=False)), # (B, F1, C, L)
            ('bn1', BatchNorm2d(F1)),
            ('dconv', Conv2d(F1, D*F1, (C, 1), bias=False, groups=F1)), # (B, D*F1, 1, L)
            ('bn2', BatchNorm2d(D*F1)),
            ('elu', ELU()),
            ('avgpool', AvgPool2d(1, 4)), # (B, D*F1, 1, L//4)
            ('dropout', Dropout(self.dropout_rate))
        ]))
        self.block2 = Sequential(OrderedDict([
            ('sconv_d', Conv2d(D*F1, D*F1, (1, sampling_rate//8), padding='same', bias=False, groups=D*F1)), # (B, D*F1, 1, L//4)
            ('sconv_p', Conv2d(D*F1, F2, (1, 1), padding='same', bias=False)), # (B, F2, 1, L//4)
            ('bn', BatchNorm2d(F2)),
            ('elu', ELU()),
            ('avgpool', AvgPool2d(1, 8)), # (B, F2, 1, L//32)
            ('dropout', Dropout(self.dropout_rate)),
            ('flatten', Flatten()) # (B, F2*L//32)
        ]))
        self.clf = Linear(F2*L//32, N) # (B, N)

    def forward(self,x:torch.Tensor):
        """
            Args:
                x: input tensor, shape (B, C, L)
            Returns:
                y: logits shape (B, N)
        """
        x = x.unsqueeze(1) # (B, 1, C, L)
        x = self.block1(x) # (B, D*F1, 1, L//4)
        x = self.block2(x) # (B, F2*L//32)
        y = self.clf(x) # (B, N)
        return y

def _constraint_linear_max_norm(linear, max_norm=0.25):
    with torch.no_grad():
        norm = linear.weight.norm().clamp(min=max_norm/2)
        desired = norm.clamp(max=max_norm)
        linear.weight *= (desired / norm)

def _constraint_filter_max_norm(conv, max_norm=1):
    with torch.no_grad():
        norm = conv.weight.norm(dim=2,keepdim=True).clamp(min=max_norm/2)
        desired = norm.clamp(max=max_norm)
        conv.weight *= (desired / norm)

def train(model, device, dataloader, optimizer, wandb, class_names, criterion=CrossEntropyLoss(), epoch=None):
    model.train()
    targets = []
    preds = []
    correct_samples = 0
    total_samples = 0
    for i, (x, cl, sj) in tqdm(enumerate(dataloader), total=len(dataloader)):
        x = x.to(device, dtype=torch.float32)
        target = cl.to(device)
        logit = model(x)
        loss = criterion(logit, target)
        pred = logit.argmax(axis=-1)
        accuracy = (pred == target).float().mean()
        targets += target.tolist()
        preds += pred.tolist()

        correct_samples += (pred == target).sum().item()
        total_samples += target.size(0)

        wandb.log({"train loss": loss.item(), "train accuracy": accuracy.item()})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _constraint_filter_max_norm(model.block1.dconv)
        _constraint_linear_max_norm(model.clf)
    if epoch is not None and epoch % 5 == 0:
        epoch_accuracy = correct_samples / total_samples
        print(f"Train accuracy after epoch {epoch}: {epoch_accuracy:.4f}")
        
        precision = precision_score(targets, preds, average='macro')
        recall = recall_score(targets, preds, average='macro')
        print(f'Precision after epoch {epoch}: {precision}')
        print(f'Recall after epoch {epoch}: {recall}')
    wandb.log({"train conf mat": wandb.plot.confusion_matrix(
        preds=preds, y_true=targets, class_names=class_names)})

def val(model, device, dataloader, wandb, class_names, epoch, config, criterion=CrossEntropyLoss()):
    model.eval()
    val_losses = []
    targets = []
    preds = []
    correct_samples_val = 0
    total_samples_val = 0
    for i, (x, cl, sj) in tqdm(enumerate(dataloader), total=len(dataloader)):
        x = x.to(device, dtype=torch.float32)
        target = cl.to(device)
        logit = model(x)
        loss = criterion(logit, target)
        val_losses.append(loss.item())
        pred = logit.argmax(axis=-1)
        accuracy = (pred == target).float().mean()
        targets += target.tolist()
        preds += pred.tolist()

        correct_samples_val += (pred == target).sum().item()
        total_samples_val += target.size(0)

        wandb.log({"val loss": loss.item(), "val accuracy": accuracy.item()})

    avg_val_loss = np.mean(val_losses)
    epoch_accuracy_val = correct_samples_val / total_samples_val
    wandb.log({"epoch": epoch, "avg_val_loss": avg_val_loss, "epoch_val_accuracy": epoch_accuracy_val})

    if epoch % 5 == 0:
        epoch_accuracy_val = correct_samples_val / total_samples_val
        print(f"Validation accuracy after epoch {epoch}: {epoch_accuracy_val:.4f}")

    if epoch % 10 == 0:
        conf_mat = confusion_matrix(targets, preds)
        precision = precision_score(targets, preds, average='macro')
        precision = precision_score(targets, preds, average=None)
        recall = recall_score(targets, preds, average=None)
        accuracy = np.diag(conf_mat).sum() / conf_mat.sum()

        new_mat = np.zeros((5, 5))
        new_mat[:4, :4] = conf_mat
        new_mat[4, :4] = precision
        new_mat[:4, 4] = recall
        new_mat[4, 4] = accuracy

        row_labels = class_names + ['Precision']
        col_labels = class_names + ['Recall']

        current_cmap = sns.color_palette("rocket", as_cmap=True)
        reversed_cmap = current_cmap.reversed()
        plt.figure(figsize=(10,10))
        ax = sns.heatmap(new_mat, annot=True, fmt=".2f", square=True, cmap=reversed_cmap, xticklabels=col_labels, yticklabels=row_labels, vmax=65, annot_kws={"size": 12, "weight": "bold"})
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, fontweight='bold')

        plt.ylabel('Actual label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted label', fontsize=12, fontweight='bold')
        plt.title(f'Confusion matrix for epoch: {epoch}', fontsize=14, fontweight='bold')

        formatted_learning_rate = f"{config['LEARNING_RATE']:.4f}".replace('.', 'p')
        directory = f"./eegnet/CM_{config['SETTING']}_{config['SEED']}_{config['EPOCHS']}_{formatted_learning_rate}_val/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(directory + f'conf_mat_epoch_{epoch}.png')
        plt.close()
    wandb.log({"val conf mat": wandb.plot.confusion_matrix(preds=preds, y_true=targets, class_names=class_names)})
    return avg_val_loss

def save_checkpoint(epoch, config, model, savepath):
    device = next(model.parameters()).device
    model.to('cpu')
    torch.save({
        'epoch': epoch,
        'config': {k:v for k,v in config.items()},
        'sampling_rate': model.sampling_rate,
        'N': model.N,
        'L': model.L,
        'C': model.C,
        'F1': model.F1,
        'D': model.D,
        'F2': model.F2,
        'dropout_rate': model.dropout_rate,
        'model_state_dict': model.state_dict()
    }, savepath)
    model.to(device)

def load_checkpoint(savepath, device):
    checkpoint = torch.load(savepath)
    epoch = checkpoint['epoch']
    config = checkpoint['config']
    model = EEGNet(
        checkpoint['sampling_rate'],
        checkpoint['N'],
        checkpoint['L'],
        checkpoint['C'],
        checkpoint['F1'],
        checkpoint['D'],
        checkpoint['F2'],
        checkpoint['dropout_rate'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return epoch, config, model

def run(model, device, train_dl, val_dl, optimizer, config, wandb):
    class_names = ["Left", "Right", "Feet", "Tongue"] if config['DATA'] == "bciciv" else ["Non-target", "Target"]
    min_val_loss = float('inf')
    min_val_loss_path = None  
    for epoch in range(1, 1 + config['EPOCHS']):
        train(model, device, train_dl, optimizer, wandb, class_names, epoch=epoch)
        avg_val_loss = val(model, device, val_dl, wandb, class_names, epoch=epoch, config=config)

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss

            if min_val_loss_path is not None and os.path.exists(min_val_loss_path):
                os.remove(min_val_loss_path)

            formatted_loss = f"{min_val_loss:.4f}".replace('.', 'p')
            min_val_loss_path = f"eegnet/checkpoints/eegnet_{config['SEED']}_{epoch}_min_{formatted_loss}.pch"
            save_checkpoint(epoch, config, model, min_val_loss_path)

    final_epoch_path = f"eegnet/checkpoints/eegnet_{config['SEED']}_{epoch}_final.pch"
    save_checkpoint(epoch, config, model, final_epoch_path)













