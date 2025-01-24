import numpy as np
import torch
from tqdm import tqdm
from scipy.linalg import sqrtm

def compute_sfid(model, test_dl, gen_dl, device, nb_test, nb_gen, eps=1e-6):
    act_test = get_activations(model, device, test_dl, nb_test, 'block1')
    act_g_v = get_activations(model, device, gen_dl, nb_gen, 'block1')
    
    # Reshape the activations to 2D arrays
    act_test = act_test.reshape(act_test.shape[0], -1)
    act_g_v = act_g_v.reshape(act_g_v.shape[0], -1)
    
    _s = min(act_test.shape[0], act_g_v.shape[0])
    act_test = act_test[:_s]
    act_g_v = act_g_v[:_s]
    
    mu_test, mu_g_v = np.mean(act_test, axis=0), np.mean(act_g_v, axis=0)
    sigma_test, sigma_g_v = np.cov(act_test, rowvar=False), np.cov(act_g_v, rowvar=False)
    
    diff = mu_test - mu_g_v
    cov_mean_sqrt, _ = sqrtm(np.dot(sigma_test, sigma_g_v), disp=False)

    if not np.isfinite(cov_mean_sqrt).all():
        offset = np.eye(sigma_test.shape[0]) * eps
        cov_mean_sqrt = sqrtm(np.dot(sigma_test + offset, sigma_g_v + offset))

    if np.iscomplexobj(cov_mean_sqrt):
        cov_mean_sqrt = cov_mean_sqrt.real

    return np.dot(diff, diff) + np.trace(sigma_test) + np.trace(sigma_g_v) - 2 * np.trace(cov_mean_sqrt)

def get_activations(model, device, dataloader, nb_samples, layer_name):
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.eval()
    handle = getattr(model, layer_name).register_forward_hook(get_activation(layer_name))
    activations = []
    
    with torch.no_grad():
        for i, (x, _, _) in tqdm(enumerate(dataloader)):
            x = x.to(device, dtype=torch.float32)
            _ = model(x)
            out = activation[layer_name]
            activations.append(out.cpu().numpy())
            
    handle.remove()
    activations = np.concatenate(activations, axis=0)
    return activations

if __name__ == "__main__":
    # model = YourModelHere()
    # test_dl = DataLoader(test_dataset, batch_size=32)
    # gen_dl = DataLoader(gen_dataset, batch_size=32)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # sfid_score = compute_sfid(model, test_dl, gen_dl, device, len(test_dataset), len(gen_dataset))
    # print("sFID Score:", sfid_score)
    pass