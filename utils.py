import torch
import numpy as np
import os
import shutil
from datasets import LandmarksDataset_flow
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def marginal_prob_std(t, sigma, device = 'cuda'):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:    
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.  
    
    Returns:
        The standard deviation.
    """    
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma, device = 'cuda'):
    """Compute the diffusion coefficient of our SDE.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.
    
    Returns:
        The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device=device)


def save_checkpoint(state, is_best, folder="./", filename="checkpoint.pth.tar"):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, filename), os.path.join(folder, "model_best.pth.tar")
        )


def plot_samples(x_samples, y_samples, epoch):
    """
    plotting code to look at both original data and model samples
    """
    fig, axs = plt.subplots(4, 4)
    fig.suptitle("Generated Samples at Epoch {}".format(epoch))
    for i in range(4):
        for j in range(4):
            cnt = 4 * i + j
            x, y = x_samples[cnt, :], y_samples[cnt, :]
            # axs[i, j].plot(x, y)
            axs[i, j].scatter(x, y, s = 10)
            

    # despine then save plot
    sns.despine()
    plt.tight_layout()
    plt.savefig('flow_imgs/samples_epoch{}.png'.format(epoch))


# def make_halfmoon_toy_dataset(n_samples=30000, batch_size=100):
#     # lucky number
#     rng = np.random.RandomState(777)

#     # generate data and normalize to 0 mean
#     data = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.05)[0]
#     data = data.astype("float32")
#     data = StandardScaler().fit_transform(data)

#     # turn this into a torch dataset
#     data = torch.from_numpy(data).float()

#     # change this to train/val/test split
#     p_idx = np.random.permutation(n_samples)
#     train_idx = p_idx[0:24000]
#     val_idx = p_idx[24000:27000]
#     test_idx = p_idx[27000:]

#     # partition data into train/valid/test
#     train_dataset = torch.utils.data.TensorDataset(data[train_idx])
#     val_dataset = torch.utils.data.TensorDataset(data[val_idx])
#     test_dataset = torch.utils.data.TensorDataset(data[test_idx])

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True
#     )
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=batch_size, shuffle=False
#     )
#     test_loader = torch.utils.data.DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False
#     )

#     return train_loader, val_loader, test_loader


def make_landmarks_dataset(batch_size=100):
    # lucky number
    rng = np.random.RandomState(777)

    # generate data
    train_dataset = LandmarksDataset_flow(mode = 'train')
    val_dataset = LandmarksDataset_flow(mode = 'test')
    test_dataset = LandmarksDataset_flow(mode = 'val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = True, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = True, num_workers = 4)

    return train_loader, val_loader, test_loader
