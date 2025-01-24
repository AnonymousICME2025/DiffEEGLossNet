from tqdm import tqdm
from eval import *
from data.utils import *
from eegnet.torch_eegnet import *
from eval.spatial_frechet_inception_distance import compute_sfid
from eval.precision_recall import compute_precision_recall  

import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("eval/conf.json",'r') as fconf:    # Open and read the configuration file
    conf = json.load(fconf)    # Load the configuration file content into a Python dictionary

_, config, model = load_checkpoint(conf['CHECKPOINT'], device)    # Load the pretrained model
model.to(device)    # Move the model to the specified device (GPU or CPU)

gen_ds = GenDataset(conf['DATA'], conf['GEN_RUN'])    # Create the generated dataset
if conf['DATA'] == 'bciciv2b':    # If the data is '2B' type
    val_ds = BCICIV2bDataset(conf['N_SUBJECTS'],True,partition='val')    # Create validation dataset
    test_ds = BCICIV2bDataset(conf['N_SUBJECTS'],True,partition='test')    # Create test dataset
else:    # If the data is '2A' type
    val_ds = BCICIV2aDataset(conf['N_SUBJECTS'],True,partition='val')    # Create validation dataset
    test_ds = BCICIV2aDataset(conf['N_SUBJECTS'],True,partition='test')    # Create test dataset
    train_ds = BCICIV2aDataset(conf['N_SUBJECTS'],True,partition='train')

gen_dl = DataLoader(gen_ds,batch_size=conf['BATCH_SIZE'],shuffle=False)    # Create data loader for generated data
val_dl = DataLoader(val_ds,batch_size=conf['BATCH_SIZE'],shuffle=False)    # Create data loader for validation data
test_dl = DataLoader(test_ds,batch_size=conf['BATCH_SIZE'],shuffle=False)    # Create data loader for test data
train_dl = DataLoader(train_ds,batch_size=conf['BATCH_SIZE'],shuffle=False)    # Create data loader for train data

# Compute IS
s, ss = compute_is(model, device, test_dl)   # Compute Inception Score for test data
print(f"Inception score - test: {s}")   	 # Print Inception Score for test data
s, ss = compute_is(model, device, gen_dl)    # Compute Inception Score for generated data
print(f"Inception score - {conf['GEN_RUN']}: {s}")    # Print Inception Score for generated data

# Compute FID
fid_tv = compute_fid(model, test_dl, val_dl, device, len(test_ds), len(val_ds))    # Compute Fréchet Inception Distance between validation and test data
print(f"FID score - test vs. val: {fid_tv}")    # Print FID score for test vs. validation data
fid_tg = compute_fid(model, test_dl, gen_dl, device, len(test_ds), len(gen_ds))    # Compute FID score between test and generated data
print(f"FID score - test vs. {conf['GEN_RUN']}: {fid_tg}")    # Print FID score for test vs. generated data
fid_tt = compute_fid(model, test_dl, train_dl, device, len(test_ds), len(train_ds))    # Compute FID score between test and train data
print(f"FID score - test vs. train: {fid_tt}")    # Print FID score for test vs. train data

# Compute sFID
sfid_tv = compute_sfid(model, test_dl, val_dl, device, len(test_ds), len(val_ds))    # Compute sFID score between validation and test data
print(f"sFID score - test vs. val: {sfid_tv}")    # Print sFID score for test vs. validation data
sfid_tg = compute_sfid(model, test_dl, gen_dl, device, len(test_ds), len(gen_ds))    # Compute sFID score between test and generated data
print(f"sFID score - test vs. {conf['GEN_RUN']}: {sfid_tg}")    # Print sFID score for test vs. generated data

# Compute Precision and Recall
precision, recall = compute_precision_recall(model, val_dl, gen_dl, k=3, num_samples=len(val_ds))  # Compute precision and recall
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")   # Print precision and recall values


