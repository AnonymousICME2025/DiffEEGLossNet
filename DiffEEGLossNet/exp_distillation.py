from diffusion.distillation import *
from diffusion.diffusion import *
from data.utils import *
from diffusion.diffeeglossnet import *
from torch.utils.data import random_split
import numpy as np
import wandb
import json
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Define a function to load the saved model state
def load_checkpoint(savepath, device):
    checkpoint = torch.load(savepath)  # Load the saved model state
    epoch = checkpoint['epoch']  # Get the saved epoch
    config = checkpoint['config']  # Get the saved model config
    function_approximator = DiffEEGLossNet(  # Create the DiffEEGLossNet model
        checkpoint['n_class'],
        checkpoint['n_subject'],
        checkpoint['N'],
        checkpoint['n'],
        checkpoint['C'],
        checkpoint['E'],
        checkpoint['K']
    )
    model = Diffusion(function_approximator, checkpoint['T'])  # Create the Diffusion model
    model.load_state_dict(checkpoint['model_state_dict'])  # Load model parameters
    model.to(device)  # Move the model to the specified device (GPU or CPU)
    return epoch, config, model  # Return the epoch, config, and model


print("Initialize")  # Print "Initialize" to indicate the start of initialization
with open("diffusion/distillation_conf.json", 'r') as fconf:  # Open the configuration file
    conf = json.load(fconf)  # Load the content of the configuration file

wandb.init(project="amal_diffusion", entity="amal_2223", config=conf)  # Initialize wandb with the project name, entity, and config
random_seed = np.random.choice(9999)  # Randomly choose a seed
config = wandb.config  # Get the wandb config
config.SEED = random_seed  # Set the seed in the wandb config
torch.manual_seed(config.SEED)  # Set the seed for PyTorch
np.random.seed(config.SEED)  # Set the seed for numpy
torch.backends.cudnn.deterministic = True  # Set PyTorch's cudnn to deterministic mode for reproducibility

print("Loading data")  # Print "Loading data" to indicate the start of data loading
# Based on the DATA field in the config, load the corresponding dataset and create the corresponding model
if config.DATA == 'BCICIV2B':
    train_ds = BCICIV2bDataset(config.N_SUBJECTS, True, partition='train')  # Load BCICIV2b training dataset
    val_ds = BCICIV2bDataset(config.N_SUBJECTS, True, partition='val')  # Load BCICIV2b validation dataset
    test_ds = BCICIV2bDataset(config.N_SUBJECTS, True, partition='test')  # Load BCICIV2b test dataset
    SIGNAL_LENGTH = 448  # Set signal length
else:
    train_ds = BCICIV2aDataset(config.N_SUBJECTS, True, partition='train')  # Load BCICIV2a training dataset
    val_ds = BCICIV2aDataset(config.N_SUBJECTS, True, partition='val')  # Load BCICIV2a validation dataset
    test_ds = BCICIV2aDataset(config.N_SUBJECTS, True, partition='test')  # Load BCICIV2a test dataset
    SIGNAL_LENGTH = 448  # Set signal length
    
# Load the pre-trained model
teacher_path = Path(f"{os.path.dirname(os.path.abspath(__file__))}/diffusion/checkpoints/diffusion_{config.TEACHER}.pch")
_, _, model = load_checkpoint(teacher_path, device)

# Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), config.LEARNING_RATE)
wandb.watch(model, log="all")  # Use wandb to monitor the model during training

# Create data loaders
train_dl = DataLoader(train_ds, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)

print("Distilling")  # Print "Distilling" to indicate the start of distillation
distill(model, device, train_dl, val_dl, optimizer, config, wandb)  # Perform model distillation


