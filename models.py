#@title Defining a time-dependent score-based model (double click to expand or collapse)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from losses import loss_fn, loss_fn_MY
from utils import *

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


torch.autograd.set_detect_anomaly(True)

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=10.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
  
  def forward(self, x, t): 
    print('x', x.shape)  
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))    
    print('embed: ', embed.shape)
    # Encoding path
    h1 = self.conv1(x)   
    print('h1: ', h1.shape) 
    print('noise:', self.dense1(embed).shape)
    ## Incorporate information from t
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)
    print('h4: ', h4.shape) 
    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.act(h)
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    print('h:', h.shape)
    print('h1s:', h1.shape)
    h = self.tconv1(torch.cat([h, h1], dim=1))
    print('h again:', h.shape)
    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]
    return h

class ScoreNet_landmarks(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, neurons=[200, 150, 100, 50], n_landmarks = 70, embed_dim=24):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.linear1 = nn.Linear(n_landmarks * 2, neurons[0])
    self.noise_dense1 = nn.Linear(embed_dim, neurons[0]) 
    self.linear2 = nn.Linear(neurons[0], neurons[1])
    self.noise_dense2 = nn.Linear(embed_dim, neurons[1]) 
    self.linear3 = nn.Linear(neurons[1], neurons[2])
    self.noise_dense3 = nn.Linear(embed_dim, neurons[2])
    self.linear4 = nn.Linear(neurons[2], neurons[3])
    self.noise_dense4 = nn.Linear(embed_dim, neurons[3]) 

    # Decoding layers where the resolution increases
    self.tlinear5 = nn.Linear(neurons[3], neurons[2])
    self.noise_dense5 = nn.Linear(embed_dim, neurons[2]) 
    self.tlinear6 = nn.Linear(neurons[2], neurons[1])
    self.noise_dense6 = nn.Linear(embed_dim, neurons[1]) 
    self.tlinear7 = nn.Linear(neurons[1], neurons[0])
    self.noise_dense7 = nn.Linear(embed_dim, neurons[0]) 
    self.tlinear8 = nn.Linear(neurons[0], n_landmarks * 2)
    self.noise_dense8 = nn.Linear(embed_dim, n_landmarks * 2) 

    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
  
  def forward(self, x, t): 
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))    
   
    h1 = self.linear1(x)
    h1 += self.noise_dense1(embed)
    # print('***** h1', h1.shape)
    ## Group normalization
    # h1 = self.gnorm1(h1) # TODO: consider normalization
    h1 = self.act(h1)
    
    h2 = self.linear2(h1)
    h2 += self.noise_dense2(embed)
    h2 = self.act(h2)
    h3 = self.linear3(h2)
    h3 += self.noise_dense3(embed)
    h3 = self.act(h3)
    h4 = self.linear4(h3)
    h4 += self.noise_dense4(embed)
    h4 = self.act(h4)

    # Decoding path
    h = self.tlinear5(h4)
    ## Skip connection from the encoding path
    h += self.noise_dense5(embed)
    h = self.act(h)

    h = self.tlinear6(h)
    h += self.noise_dense6(embed)
    h = self.act(h)
    h = self.tlinear7(h)
    h += self.noise_dense7(embed)
    h = self.act(h)
    h = self.tlinear8(h)
    h += self.noise_dense8(embed)
    h = self.act(h)


    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None]
    return h


class MaskedLinear(nn.Linear):
    """Masked linear layer for MADE: takes in mask as input and masks out connections in the linear layers."""

    def __init__(self, input_size, output_size, mask):
        super().__init__(input_size, output_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class PermuteLayer(nn.Module):
    """Layer to permute the ordering of inputs.

    Because our data is 2-D, forward() and inverse() will reorder the data in the same way.
    """

    def __init__(self, num_inputs):
        super(PermuteLayer, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])

    def forward(self, inputs):
        return inputs[:, self.perm], torch.zeros(
            inputs.size(0), 1, device=inputs.device
        )

    def inverse(self, inputs):
        return inputs[:, self.perm], torch.zeros(
            inputs.size(0), 1, device=inputs.device
        )


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation.
    https://arxiv.org/abs/1502.03509

    Uses sequential ordering as in the MAF paper.
    Gaussian MADE to work with real-valued inputs"""

    def __init__(self, input_size, hidden_size, n_hidden):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden

        masks = self.create_masks()

        # construct layers: inner, hidden(s), output
        self.net = [MaskedLinear(self.input_size, self.hidden_size, masks[0])]
        self.net += [nn.ReLU(inplace=True)]
        # iterate over number of hidden layers
        for i in range(self.n_hidden):
            self.net += [MaskedLinear(self.hidden_size, self.hidden_size, masks[i + 1])]
            self.net += [nn.ReLU(inplace=True)]
        # last layer doesn't have nonlinear activation
        self.net += [
            MaskedLinear(self.hidden_size, self.input_size * 2, masks[-1].repeat(2, 1))
        ]
        self.net = nn.Sequential(*self.net)

    def create_masks(self):
        """
        Creates masks for sequential (natural) ordering.
        """
        masks = []
        input_degrees = torch.arange(self.input_size)
        degrees = [input_degrees]  # corresponds to m(k) in paper

        # iterate through every hidden layer
        for n_h in range(self.n_hidden + 1):
            degrees += [torch.arange(self.hidden_size) % (self.input_size - 1)]
        degrees += [input_degrees % self.input_size - 1]
        self.m = degrees

        # output layer mask
        for (d0, d1) in zip(degrees[:-1], degrees[1:]):
            masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

        return masks

    def forward(self, z):
        """
        Run the forward mapping (z -> x) for MAF through one MADE block.
        :param z: Input noise of size (batch_size, self.input_size)
        :return: (x, log_det). log_det should be 1-D (batch_dim,)
        """
        x = torch.zeros_like(z)

        # YOUR CODE STARTS HERE
        # raise NotImplementedError
        for i in range(self.input_size):
            mu_alpha = self.net(x.clone())
            mu, alpha = torch.chunk(mu_alpha, 2, dim = 1)
            x[:, i] = mu[:, i] + z[:,i] * torch.exp(alpha[:,i])
            x[:, i] = mu[:, i] + z[:,i] * torch.exp(alpha[:,i])


        log_det = -torch.sum(alpha, dim = 1)
        # YOUR CODE ENDS HERE

        return x, log_det

    def inverse(self, x):
        """
        Run one inverse mapping (x -> z) for MAF through one MADE block.
        :param x: Input data of size (batch_size, self.input_size)
        :return: (z, log_det). log_det should be 1-D (batch_dim,)
        """
        # YOUR CODE STARTS HERE
        # raise NotImplementedError
        mu_alpha = self.net(x)
        mu, alpha = torch.chunk(mu_alpha, 2, dim = 1)
        z = (x - mu) / torch.exp(alpha)
        log_det = -torch.sum(alpha, dim = 1)

        # YOUR CODE ENDS HERE

        return z, log_det


class MAF(nn.Module):
    """
    Masked Autoregressive Flow, using MADE layers.
    https://arxiv.org/abs/1705.07057
    """

    def __init__(self, input_size, hidden_size, n_hidden, n_flows):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.n_flows = n_flows
        self.base_dist = torch.distributions.normal.Normal(0, 1)

        # need to flip ordering of inputs for every layer
        nf_blocks = []
        for i in range(self.n_flows):
            nf_blocks.append(MADE(self.input_size, self.hidden_size, self.n_hidden))
            nf_blocks.append(PermuteLayer(self.input_size))  # permute dims
        self.nf = nn.Sequential(*nf_blocks)

    def log_probs(self, x):
        """
        Obtain log-likelihood p(x) through one pass of MADE
        :param x: Input data of size (batch_size, self.input_size)
        :return: log_prob. This should be a Python scalar.
        """
        # YOUR CODE STARTS HERE
        # raise NotImplementedError
        log_det_sum = None
        for layer in self.nf:
            x, log_det = layer.inverse(x)
            if isinstance(layer, MADE):
                log_det_sum =  log_det_sum + log_det if log_det_sum  is not None else log_det
        # z = x
        log_probs = -0.5 * (np.log(2 * np.pi) * self.input_size + (x ** 2).sum(dim = 1))
        # print('-----', log_probs.mean(), log_det_sum.mean())
        log_probs += log_det_sum
        log_prob = torch.mean(log_probs)
        # YOUR CODE ENDS HERE

        return log_prob

    def loss(self, x):
        """
        Compute the loss.
        :param x: Input data of size (batch_size, self.input_size)
        :return: loss. This should be a Python scalar.
        """
        return -self.log_probs(x)

    def sample(self, device, n):
        """
        Draw <n> number of samples from the model.
        :param device: [cpu,cuda]
        :param n: Number of samples to be drawn.
        :return: x_sample. This should be a numpy array of size (n, self.input_size)
        """
        with torch.no_grad():
            x_sample = torch.randn(n, self.input_size).to(device)
            for flow in self.nf[::-1]:
                x_sample, log_det = flow.forward(x_sample)
            x_sample = x_sample.view(n, self.input_size)
            x_sample = x_sample.cpu().data.numpy()
        x_idx = np.array([i for i in range(x_sample.shape[1]) if i % 2 ==0])
        y_idx = np.array([i + 1 for i in x_idx])
        return x_sample[:, x_idx], x_sample[:, y_idx]





if __name__ == '__main__':
    # test 
    device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

    
    sigma =  25.0#@param {'type':'number'}
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
    


    score_model = torch.nn.DataParallel(ScoreNet_landmarks(marginal_prob_std=marginal_prob_std_fn, n_landmarks = 68))
    score_model = score_model.to(device)
    # x = torch.randn((17,1,28,28))
    x = torch.randn((17,136))
    x = x.to(device) 
    loss = loss_fn_MY(score_model, x, marginal_prob_std_fn)
