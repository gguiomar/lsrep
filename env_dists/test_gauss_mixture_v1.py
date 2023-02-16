#%% 
import torch
import numpy as np
import normflows as nf
from sklearn import cluster, datasets, mixture
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
from matplotlib import pyplot as plt
from tqdm import tqdm

import grid_gaussian as gg

# define two symmetric gaussian means and covariances 
means = torch.tensor([[-3, 0], [0, 3]], dtype=torch.float)
weights = torch.tensor([0.5, 0.5], dtype=torch.float)

q0 = gg.GaussianMixture(n_modes = 2, dim = 2, loc=means, weights=weights, trainable=True)

# plot a 2d histogram of q0

num_samples = 10000
z, _ = q0.forward(num_samples=num_samples)
z = z.detach().numpy()
plt.hist2d(z[:,0], z[:,1], bins=100)
plt.show()


# %%



q0 = nf.distributions.DiagGaussian(2)

# %%
