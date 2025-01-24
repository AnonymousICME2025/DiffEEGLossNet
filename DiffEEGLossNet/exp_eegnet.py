import json
import wandb
import numpy as np
from pathlib import Path
from data.utils import *
from eegnet.torch_eegnet import *
from torch.utils.data import ConcatDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define a function to load the saved model state
def load_checkpoint(savepath, device):
    checkpoint = torch.load(savepath)  # Load the saved model state
    epoch = checkpoint['epoch']  # Get the saved epoch
    config = checkpoint['config']  # Get the saved model config
    model = EEGNet(  # Create the EEGNet model
        checkpoint['sampling_rate'],
        checkpoint['N'],
        checkpoint['L'],
        checkpoint['C'],
        checkpoint['F1'],
        checkpoint['D'],
        checkpoint['F2'],
        checkpoint['dropout_rate'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])  # Load the model parameters
    model.to(device)  # Move the model to the specified device (GPU or CPU)
    return epoch, config, model  # Return the epoch, config, and model

print("Initialize")  # Print "Initialize" to indicate the start of initialization

with open("eegnet/eegnet_conf.json", 'r') as fconf:  # Open the configuration file
    conf = json.load(fconf)  # Load the content of the configuration file

SAMPLING_RATE = {'bciciv': 128, 'bciciv2b': 128}  # Define a dictionary to store the sampling rate for different datasets
conf['SAMPLING_RATE'] = SAMPLING_RATE[conf['DATA']]  # Set the corresponding sampling rate based on the DATA field in the config

NB_CLASSES = {'bciciv': 4, 'bciciv2b': 2}  # Define a dictionary to store the number of classes for different datasets
conf['NB_CLASSES'] = NB_CLASSES[conf['DATA']]  # Set the corresponding number of classes based on the DATA field in the config

SIGNAL_LENGTH = {'bciciv': 448, 'bciciv2b': 448}  # Define a dictionary to store the signal length for different datasets
conf['SIGNAL_LENGTH'] = SIGNAL_LENGTH[conf['DATA']]  # Set the corresponding signal length based on the DATA field in the config

NB_CHANS = {'bciciv': 25, 'bciciv2b': 3}  # Define a dictionary to store the number of channels for different datasets
conf['NB_CHANS'] = NB_CHANS[conf['DATA']]  # Set the corresponding number of channels based on the DATA field in the config

if "checkpoint" in conf:  # If "checkpoint" is in the config, load the saved model state, otherwise create a new wandb experiment
    savepath = Path(conf["checkpoint"])  # Get the path of the saved model state
    epoch, cp_conf, model = load_checkpoint(savepath, device)  # Load the saved model state
    wandb.init(project="amal_diffusion", entity="amal_2223", config=cp_conf)  # Initialize the wandb experiment with project name, entity, and config
else:
    wandb.init(project="amal_diffusion", entity="amal_2223", config=conf)  # Initialize the wandb experiment with project name, entity, and config

random_seed = np.random.choice(9999)  # Randomly choose a seed
config = wandb.config  # Get the wandb config
config.SEED = random_seed  # Set the seed in the wandb config
torch.manual_seed(config.SEED)  # Set the seed for PyTorch
np.random.seed(config.SEED)  # Set the seed for numpy
torch.backends.cudnn.deterministic = True  # Set PyTorch's cudnn to deterministic mode to ensure reproducibility
wandb.define_metric("train accuracy", summary="mean")  # Define the wandb metric, in this case the mean of training accuracy
wandb.define_metric("val accuracy", summary="mean")  # Define the wandb metric, in this case the mean of validation accuracy
model = EEGNet(config.SAMPLING_RATE, config.NB_CLASSES, config.SIGNAL_LENGTH, config.NB_CHANS)  # Create the EEGNet model
model.to(device)  # Move the model to the specified device (GPU or CPU)
#  optimizer = torch.optim.Adam(model.parameters(), config.LEARNING_RATE)  # Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)  # Use L2 regularization to prevent overfitting

# Weight decay works by adding a term proportional to the size of the weights to the loss function. This encourages the model to have smaller weights, making the model simpler and reducing the risk of overfitting.
# During training, weight decay applies a small penalty to each weight, causing them to tend towards smaller values.

wandb.watch(model, log="all")  # Use wandb to monitor the training process of the model

print("Loading data")  
gen_train_ds = GenDataset(config.DATA, config.GEN_RUN)  # Create the generated training dataset  

# Load the corresponding dataset based on the DATA field in the config
if config.DATA == 'BCICIV2B':  
    src_train_ds = BCICIV2bDataset(config.N_SUBJECTS, False, partition='train')
    val_ds = BCICIV2bDataset(config.N_SUBJECTS, False, partition='val')
    test_ds = BCICIV2bDataset(config.N_SUBJECTS, False, partition='test')
else:  # Otherwise, load the BCICIV2a dataset
    src_train_ds = BCICIV2aDataset(config.N_SUBJECTS, True, partition='train')
    val_ds = BCICIV2aDataset(config.N_SUBJECTS, True, partition='val')
    test_ds = BCICIV2aDataset(config.N_SUBJECTS, True, partition='test')

val_dl = DataLoader(val_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)  # Create the validation data loader
test_dl = DataLoader(test_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)  # Create the test data loader


print("Training")
# Based on the SETTING field in the config, choose the corresponding training setting
if config.SETTING == 'DEFAULT':  # If the training setting is 'DEFAULT', use only the source training dataset for training
    train_ds = src_train_ds
    train_dl = DataLoader(train_ds, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)  # Create the training data loader
    run(model, device, train_dl, val_dl, optimizer, config, wandb)  # Run the model training
elif config.SETTING == 'PRETRAIN':  # If the training setting is 'PRETRAIN'
    train_ds = gen_train_ds  # First pretrain with the generated training dataset
    train_dl = DataLoader(train_ds, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)  # Create the training data loader
    run(model, device, train_dl, val_dl, optimizer, config, wandb)  # Run the model training 
    train_ds = src_train_ds  # Then train with the source training dataset
    train_dl = DataLoader(train_ds, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)  # Create the training data loader
    run(model, device, train_dl, val_dl, optimizer, config, wandb)  # Run the model training
elif config.SETTING == 'DOUBLE':  # If the training setting is 'DOUBLE'
    train_ds = ConcatDataset([src_train_ds, gen_train_ds])  # Use both the source and generated training datasets for training
    train_dl = DataLoader(train_ds, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)  # Create the training data loader
    run(model, device, train_dl, val_dl, optimizer, config, wandb)  # Run the model training
else:  # If the training setting is not 'DEFAULT', 'PRETRAIN', or 'DOUBLE', print an error message and exit
    print("SETTING must be DEFAULT, PRETRAIN or DOUBLE.")
    exit(1)




