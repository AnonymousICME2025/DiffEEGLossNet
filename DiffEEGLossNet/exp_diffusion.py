from diffusion.diffusion import *
from diffusion.eegwave import *
from data.utils import *
from torch.utils.data import random_split
import numpy as np
import wandb
import json
import os
from eval import *
from run_sampling import sample
from eegnet.torch_eegnet import EEGNet
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Define a function to load the saved model state
def load_checkpoint(savepath, device):
    checkpoint = torch.load(savepath)  # Load the saved model state
    epoch = checkpoint['epoch']  # Get the saved epoch
    config = checkpoint['config']  # Get the saved model config
    function_approximator = EEGWave(  # Create EEGWave model
        checkpoint['n_class'],
        checkpoint['n_subject'],
        checkpoint['N'],
        checkpoint['n'],
        checkpoint['C'],
        checkpoint['E'],
        checkpoint['K']
    )
    model = Diffusion(function_approximator, checkpoint['T'])  # Create Diffusion model
    model.load_state_dict(checkpoint['model_state_dict'])  # Load model parameters
    model.to(device)  # Move model to the specified device (GPU or CPU)
    return epoch, config, model  # Return epoch, config, and model


print("Initialize")  # Print "Initialize" to indicate the start of initialization
with open("diffusion/diffusion_conf.json", 'r') as fconf:  # Open the configuration file
    conf = json.load(fconf)  # Load the configuration file content


# If the "checkpoint" field exists in the config, load the saved model state, otherwise create a new wandb experiment
if "checkpoint" in conf:
    savepath = Path(conf["checkpoint"])
    epoch, cp_conf, model = load_checkpoint(savepath, device)
    wandb.init(project="amal_diffusion", entity="amal_2223", config=cp_conf)
else:
    wandb.init(project="amal_diffusion", entity="amal_2223", config=conf)

# Set random seed for reproducibility
random_seed = np.random.choice(9999)
config = wandb.config
config.SEED = random_seed
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
torch.backends.cudnn.deterministic = True  # Set PyTorch cudnn to deterministic mode for reproducibility

print("Loading data")  # Print "Loading data" to indicate the start of data loading

# Load the corresponding dataset based on the DATA field in the config and create the corresponding model
if config.DATA == 'BCICIV2B':
    train_ds = BCICIV2bDataset(config.N_SUBJECTS, True, partition='train')  # Modify True-False
    val_ds = BCICIV2bDataset(config.N_SUBJECTS, True, partition='val')
    test_ds = BCICIV2bDataset(config.N_SUBJECTS, True, partition='test')
    model = Diffusion(EEGWave(n_class=2, n_subject=18, E=64))  # Create Diffusion model  # Modify 70-64
    SIGNAL_LENGTH = 512  # Set signal length
else:
    train_ds = BCICIV2aDataset(config.N_SUBJECTS, True, partition='train')  # Modify True-False
    val_ds = BCICIV2aDataset(config.N_SUBJECTS, True, partition='val')
    test_ds = BCICIV2aDataset(config.N_SUBJECTS, True, partition='test')
    model = Diffusion(EEGWave(n_class=4, n_subject=9, E=25))  # Create Diffusion model
    SIGNAL_LENGTH = 448

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), config.LEARNING_RATE)  # Create optimizer
wandb.watch(model, log="all")  # Use wandb to monitor model training
train_dl = DataLoader(train_ds, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)  # Create training data loader
val_dl = DataLoader(val_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)  # Create validation data loader
test_dl = DataLoader(test_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)  # Create test data loader


# Define a function to load EEGNet model state
def load_checkpoint(savepath, device):
    checkpoint = torch.load(savepath)
    epoch = checkpoint['epoch']
    config = checkpoint['config']
    model = EEGNet(  # Create EEGNet model
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


# Load EEGNet model state
_, conf_eeg, eegnet = load_checkpoint("eegnet/checkpoints/eegnet_4434_800_testBest.pch", device)

eegnet.to(device)

# Compute Inception Score and Fréchet Inception Distance
is_t, _ = compute_is(eegnet, device, test_dl)
fid_t = compute_fid(eegnet, test_dl, val_dl, device, len(test_ds), len(val_ds))

# Define sampling configuration
sampling_conf = {
    "checkpoint": None,
    "nb_samples": 36,
    "data": config.DATA.lower(),
    "set": 1,
    "signal_length": SIGNAL_LENGTH,
    "gamma": 0.1
}

# Define a function to run model training
def run(model, device, train_dl, val_dl, optimizer, config, wandb):
    for epoch in range(1, 1 + config.EPOCHS):  # For each training epoch
        train(model, device, train_dl, optimizer, wandb, config.CLASS_CONDITIONING, config.SUBJECT_CONDITIONING)  # Perform training
        val(model, device, val_dl, wandb, config.CLASS_CONDITIONING, config.SUBJECT_CONDITIONING)  # Perform validation
        savepath = Path(f"diffusion/checkpoints/diffusion_{config.SEED}_{epoch}.pch")  # Define path to save model state
        save_checkpoint(epoch, config, model, savepath)  # Save model state

        # If the current epoch is a multiple of 16, perform sampling and compute Inception Score and FID
        if epoch % 16 == 0:
            sampling_conf["checkpoint"] = f"diffusion_{config.SEED}_{epoch}.pch"
            with open("diffusion/sampling_conf.json", 'w') as f:  # Write sampling configuration to file
                json.dump(sampling_conf, f)  # Perform sampling
            sample_path = sample()

            gen_ds = GenDataset(config.DATA.lower(), os.path.basename(sample_path))  # Create generated dataset
            gen_dl = DataLoader(gen_ds, batch_size=sampling_conf["nb_samples"], shuffle=False)

            is_g, _ = compute_is(eegnet, device, gen_dl)
            fid_g = compute_fid(eegnet, test_dl, gen_dl, device, len(test_ds), len(gen_ds))

            wandb.log({
                "is_t": is_t,
                "fid_t": fid_t,
                "is_g": is_g,
                "fid_g": fid_g
            })

print("Training")
run(model, device, train_dl, val_dl, optimizer, config, wandb)

