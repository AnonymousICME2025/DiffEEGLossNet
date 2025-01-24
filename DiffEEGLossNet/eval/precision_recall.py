import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_precision_recall(model: torch.nn.Module, real_dl, gen_dl, k=3, num_samples=50000, eps=1e-6):
    """
    Compute Precision and Recall between the real and generated samples.
    
    Args:
        model: The model (e.g., a classifier like EEGNet) that can extract features from data.
        real_dl: DataLoader for real samples.
        gen_dl: DataLoader for generated samples.
        k: The number of nearest neighbors for the k-NN algorithm.
        num_samples: The number of samples to use for computation (it may depend on the dataset size).
        eps: Small value to avoid division by zero in computations.
        
    Returns:
        precision: The precision score between real and generated samples.
        recall: The recall score between real and generated samples.
    """
    real_features = get_features(model, real_dl, num_samples)
    gen_features = get_features(model, gen_dl, num_samples)

    # Compute precision
    precision = compute_precision(real_features, gen_features, k)
    # Compute recall
    recall = compute_recall(real_features, gen_features, k)
    
    return precision, recall

def get_features(model, dataloader, num_samples):
    """
    Extract features from the model's intermediate layers for the given dataset.
    
    Args:
        model: The model to extract features from.
        dataloader: DataLoader for the dataset.
        num_samples: The number of samples to extract features for.
        
    Returns:
        features: Feature vectors for the dataset.
    """
    model.eval()
    features = np.zeros((num_samples, model.clf.in_features))  # Adjust to match model's output
    start_idx = 0
    with torch.no_grad():
        for i, (x, _, _) in tqdm(enumerate(dataloader)):
            x = x.to(device, dtype=torch.float32)
            output = model(x)
            features[start_idx:start_idx + x.size(0)] = output.cpu().numpy()
            start_idx += x.size(0)
    return features

def compute_precision(real_features, gen_features, k=3):
    """
    Compute the precision score based on k-NN.
    
    Args:
        real_features: Feature vectors of real samples.
        gen_features: Feature vectors of generated samples.
        k: The number of nearest neighbors.
        
    Returns:
        precision: The precision score between real and generated samples.
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(real_features)
    distances, indices = nbrs.kneighbors(gen_features)
    
    precision = np.mean(np.any(distances <= np.linalg.norm(real_features[indices], axis=-1), axis=1))
    return precision

def compute_recall(real_features, gen_features, k=3):
    """
    Compute the recall score based on k-NN.
    
    Args:
        real_features: Feature vectors of real samples.
        gen_features: Feature vectors of generated samples.
        k: The number of nearest neighbors.
        
    Returns:
        recall: The recall score between real and generated samples.
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(gen_features)
    distances, indices = nbrs.kneighbors(real_features)
    
    recall = np.mean(np.any(distances <= np.linalg.norm(gen_features[indices], axis=-1), axis=1))
    return recall

# Example usage with dataloaders for real and generated data
real_dl = DataLoader(real_dataset, batch_size=64, shuffle=False)
gen_dl = DataLoader(gen_dataset, batch_size=64, shuffle=False)

precision, recall = compute_precision_recall(model, real_dl, gen_dl, k=3, num_samples=50000)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')


