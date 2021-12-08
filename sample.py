from torchvision.utils import make_grid
import torch
import functools
from samplers import ode_sampler_MY
from utils import *
from models import *
from samplers import *
import cv2

## Load the pre-trained checkpoint from disk.
sigma =  15.0
device = 'cuda'
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

score_model = torch.nn.DataParallel(ScoreNet_landmarks(marginal_prob_std=marginal_prob_std_fn, n_landmarks = 68, embed_dim = 120))
score_model = score_model.to(device)
device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
ckpt = torch.load('weights/ckpt_999.pth', map_location=device)
score_model.load_state_dict(ckpt)

sample_batch_size = 64 
sampler = Euler_Maruyama_sampler_MY 

img_size = 256
## Generate samples using the specified sampler.
samples = sampler(score_model, 
                  marginal_prob_std_fn,
                  diffusion_coeff_fn, 
                  sample_batch_size, 
                  device=device)

print(samples.shape, samples.min(), samples.max())
samples = samples - samples.min()
samples = (samples / samples.max()) * img_size

for i in range(samples.shape[0]):
    img = np.ones((img_size, img_size, 3)) * 255
    for j in range(68):
        x, y = samples[i, j], samples[i, j + 68] 
        img = cv2.circle(img, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=3)
        cv2.imwrite('images/{}.jpeg'.format(i), img) 
# print()
# print(samples[0, :])



## Sample visualization.
# samples = samples.clamp(0.0, 1.0)
# %matplotlib inline
# import matplotlib.pyplot as plt
# sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

# plt.figure(figsize=(6,6))
# plt.axis('off')
# plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
# plt.show()