import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from models import *
from datasets import *
from tqdm import tqdm


sigma =  5.0
device = 'cuda'
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

score_model = torch.nn.DataParallel(ScoreNet_landmarks(marginal_prob_std=marginal_prob_std_fn, n_landmarks = 68, embed_dim = 120))
score_model = score_model.to(device)

n_epochs =   1000
## size of a mini-batch
batch_size =  32
## learning rate
lr=1e-4 

dataset = LandmarksDataset_68() #'.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

optimizer = Adam(score_model.parameters(), lr=lr)
# tqdm_epoch = tqdm(range(n_epochs)) # .notebook.trange(n_epochs)
losses = []
print('-' * 250)
for epoch in range(n_epochs):
  print()
  # if epoch % 10 == 0:
  #   lr /= 3
  #   optimizer = Adam(score_model.parameters(), lr=lr)
  avg_loss = 0.
  num_items = 0
  loop = tqdm(data_loader)
  for i, lmrks in enumerate(loop):
    if i % 50 == 0:
      avg_loss1 = 0
    lmrks = lmrks.to(device)    
    # loss = loss_fn(score_model, lmrks, marginal_prob_std_fn)
    loss = loss_fn_MY(score_model, lmrks, marginal_prob_std_fn)
    optimizer.zero_grad()
    loss.backward()    
    optimizer.step()
    avg_loss1 += loss.item()
    avg_loss += loss.item() * lmrks.shape[0]
    num_items += lmrks.shape[0]
    # update progress bar description
    loop.set_description('Epoch [{}/{}]'.format(epoch + 1, n_epochs))
    loop.set_postfix(loss = loss.item())
    # if i%50 - 50 == -1:
    #   losses.append(avg_loss1 / 50)
  # Print the averaged training loss so far.
  # tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
  print('Average Loss: {:5f}'.format(avg_loss / num_items))
  losses.append(avg_loss / num_items)
  # Update the checkpoint after each epoch of training.
  torch.save(score_model.state_dict(), 'weights/ckpt_{}.pth'.format(epoch))
  print()
  print('-' * 150)
with open('losses.txt', 'w') as file:
  file.write('\n'.join(str(loss) for loss in losses))