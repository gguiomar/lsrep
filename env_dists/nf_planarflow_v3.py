#%% planar flow example - checking the inner layers

# Import required packages
import torch
import numpy as np
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import tqdm

import imageio
import os
import time
from utils import *


###################### MODEL STARTS HERE ######################
# Move model on GPU if available

enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

K = 2
flows = []
## K planar flows
for i in range(K):
    flows += [nf.flows.Planar((2,))]

# # K planar flows where the middle 5 layers are radial flows
# middle_layers = [K/2-1, K/2, K/2+1]
# for i in range(K):
#     if i in middle_layers:
#         flows += [nf.flows.Radial((2,))]
#     else:
#         flows += [nf.flows.Planar((2,))]

print(flows)
coords_psa = [[-3, -1], [-2.5, -1], [-2, -1], [-1.5, -1], [-1, -1], [-0.5, -1], [0.5,1], [1, 1], [1.5, 1], [2, 1], [2.5, 1], [3, 1]]
base, target = generate_base_and_target('line', 'psa', n_grid = 4, scale_factor= 0.2, coords = coords_psa, scale_factor_psa=0.2)
plot_gm_dist(base, 10000)
plot_gm_dist(target, 10000)


nfm = nf.NormalizingFlow(q0=base, flows=flows, p=target)
nfm.to(device)

# Plot initial flow distribution
grid_size = 200
z, _ = nfm.sample(num_samples = 2 ** 20)    
z_np = z.to('cpu').data.numpy()
plt.figure(figsize=(10, 10))
plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (grid_size, grid_size), range=[[-3, 3], [-3, 3]])
plt.show()

# Train model
max_iter = 7000
num_samples = 100
anneal_iter = 100
annealing = True # what is annealing?
show_iter = 1000
plotting = False
make_gif = True
save_fig = True

#%%

loss_hist, nfm, path, nfm_h = train_flow_model(nfm, max_iter, num_samples, anneal_iter, 
                                  annealing, show_iter, plotting, save_fig)

num_samples = 3000
xy_all = flow_layer_pass_through(base, nfm, num_samples, K) #using same base distribution as sim
range_v = [[-4, 4], [-4, 4]]
filename = 'pass_through_layers'

if K > 2:
    plot_flow_layers_save(xy_all, range_v, filename, path)
else:
    plot_2layer_flow_save(xy_all, range_v, path)


#%%

#num_samples = 10000
means, weights =  grid_gaussian_params(4)
num_dim  = 2 # number of dimensions
num_modes = means.shape[0]
scale_factor = 0.5
covs = scale_factor * np.ones((num_modes, num_dim))
base = nf.distributions.GaussianMixture(n_modes = num_modes, dim = num_dim, loc=means, scale = covs, weights=weights, trainable=False)

xy_all = flow_layer_pass_through(base, nfm, num_samples, K)
path = 'plots/20230220-120256/'
rs = 6
range_v = [[-rs, rs], [-rs, rs]]
filename = 'pass_through_grid_gaussian_uniform'
plot_flow_layers_save(xy_all, range_v, filename, False)

#%%

# for each iteration that was frozen as a flow object, pass the grid through the flows

means, weights =  grid_gaussian_params(4)
num_dim  = 2 # number of dimensions
num_modes = means.shape[0]
scale_factor = 0.1
covs = scale_factor * np.ones((num_modes, num_dim))

grid_gaussian_plot = nf.distributions.GaussianMixture(n_modes = num_modes, dim = num_dim, loc=means, scale = covs, weights=weights, trainable = False)
#plot_gm_dist(grid_gaussian_plot, 100000)

num_samples = 1000
flow_grid = grid_flow_pass_through(grid_gaussian_plot, nfm, num_samples, K)
rs = 6
range_v = [[-rs,rs],[-rs,rs]]

for nfm_f in nfm_h[-5:-1]:
    flow_grid = grid_flow_pass_through(grid_gaussian_plot, nfm_f, num_samples, K)    
    plot_flow_layers(flow_grid, range_v) # use a base distribution as input

#%% make gif

folder_name = 'plots/20230220-183156/' # need to automate folder name extraction
gif_name = 'gif_test2.gif'
make_gif_from_folder(folder_name, gif_name)

# %%
