#%% planar flow example - checking the inner layers

# Import required packages
import torch
import numpy as np
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import tqdm


#%%
### FUNCTIONS 

def train_flow_model(nfm, max_iter, num_samples, anneal_iter, annealing, show_iter, plotting):

    loss_hist = np.array([])
    optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-3, weight_decay=1e-4)

    grid_size = 100

    for it in tqdm(range(max_iter)):
        optimizer.zero_grad()
        if annealing:
            loss = nfm.reverse_kld(num_samples, beta=np.min([1., 0.01 + it / anneal_iter]))
        else:
            loss = nfm.reverse_kld(num_samples)
        loss.backward()
        optimizer.step()
        
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        
        if plotting:
            ## plot learned dist at each batch of iterations
            if (it + 1) % show_iter == 0:
                torch.cuda.manual_seed(0)
                z, _ = nfm.sample(num_samples=2 ** 20)
                z_np = z.to('cpu').data.numpy()
                plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (grid_size, grid_size), range=[[-3,3], [-3, 3]])
                plt.show()

    return loss_hist, nfm


# FLOW LAYERS ANALYSIS

def flow_layer_forward(nfm, layer_n, num_samples):

    q0 = nf.distributions.DiagGaussian(2)
    z, _ = q0.forward(num_samples= num_samples)

    # run the q0 samples through the flow layers and plot the results
    xy = np.zeros((len(z),2))
    for i,e in enumerate(z):
        xy[i,:] = nfm.flows[layer_n].forward(e)[0].detach().numpy()

    return xy

# plot all the layers as a 4x4 grid
def plot_flow_layers(xy_all):

    grid_size_plot = 100
    # number of plots
    n_plots = xy_all.shape[0]
    # if the number of plots is not a square number, add extra plots
    n_rows = int(np.ceil(np.sqrt(n_plots)))
    n_cols = int(np.ceil(n_plots/n_rows))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15,15))
    for i in range(n_rows):  
        for j in range(n_cols):
            if i*n_cols + j < n_plots:
                axs[i,j].hist2d(xy_all[i*n_cols + j,:,0].flatten(), xy_all[i*n_cols + j,:,1].flatten(), (grid_size_plot, grid_size_plot), range=[[-3,3], [-3, 3]])
                axs[i,j].set_title('Layer ' + str(i*n_cols + j))
                axs[i,j].set_xlim([-3,3])
                axs[i,j].set_ylim([-3,3])
            else:
                axs[i,j].axis('off')
    plt.show()

def flow_layer_pass_through(base, nfm, num_samples, n_layers):

    xy = np.zeros((n_layers, num_samples, 2))
    for k in range(n_layers):
        if k == 0:
            z, _ = base.forward(num_samples = num_samples)
        else:
            # change xy to tensor format
            z = torch.from_numpy(xy[k-1,:,:]).float()

        for i,e in enumerate(z):
            xy[k, i,:] = nfm.flows[k].forward(e)[0].detach().numpy()

    return xy

def plot_gm_dist(p, num_samples):

    grid_size = 100
    z, _ = p.forward(num_samples)
    z = z.detach().numpy()
    plt.hist2d(z[:, 0].flatten(), z[:, 1].flatten(), (grid_size, grid_size), range=[[-3,3], [-3, 3]], cmap='viridis')
    plt.show()

def plot_twomoon_dist(p, num_samples):

    grid_size = 100
    z = p.sample(num_samples)
    z = z.detach().numpy()
    plt.hist2d(z[:, 0].flatten(), z[:, 1].flatten(), (grid_size, grid_size), range=[[-3,3], [-3, 3]])
    plt.show()

def grid_gaussian_params(N):

    means = []
    for i in range(-N, N):
        for j in range(-N, N):
            means.append([i, j])

    means = torch.tensor(means, dtype=torch.float)
    weights = torch.tensor([1/len(means)] * len(means), dtype=torch.float)

    return means, weights


def plot_learned_model(nfm, num_samples):
    grid_size = 100
    z, _ = nfm.sample(num_samples=2**20)
    z_np = z.to('cpu').data.numpy()
    plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (grid_size, grid_size), range=[[-3, 3], [-3, 3]])
    plt.show()

#%%

means, weights =  grid_gaussian_params(4)
print(means, weights)

num_dim  = 2 # number of dimensions
num_modes = means.shape[0]
scale_factor = 0.4
covs = scale_factor * np.ones((num_modes, num_dim))

grid_gaussian = nf.distributions.GaussianMixture(n_modes = num_modes, dim = num_dim, loc=means, scale = covs, weights=weights, trainable=False)
plot_gm_dist(grid_gaussian, 100000)

#%%  TWO MOONS  DISTRIBUTION
two_moon = nf.distributions.TwoMoons()
#plot_twomoon_dist(two_moon, 5000)

#%% 2 gaussian distr

means = torch.tensor([[-1, 0], [1, 0]], dtype=torch.float)
weights = torch.tensor([0.5, 0.5], dtype=torch.float)
num_dim  = 2 # number of dimensions
num_modes = means.shape[0]
scale_factor = 0.4
covs = scale_factor * np.ones((num_modes, num_dim))

two_gaussians = nf.distributions.GaussianMixture(n_modes = num_modes, dim = num_dim, loc=means, scale = covs, weights=weights, trainable=False)
#plot_gm_dist(two_gaussians, 100000)

#%% n gaussian distr

means = torch.tensor([[-2, -2], [2, 2], [0, 2], [-0,-2]], dtype=torch.float)
weights = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float)

num_dim  = 2 # number of dimensions
num_modes = means.shape[0]
scale_factor = 0.3
covs = scale_factor * np.ones((num_modes, num_dim))
three_gaussians = nf.distributions.GaussianMixture(n_modes = num_modes, dim = num_dim, loc=means, scale = covs, weights=weights, trainable=False)
#plot_gm_dist(three_gaussians, 100000)


#%% 1 gaussian distr

means = torch.tensor([[0, 0]], dtype=torch.float)
weights = torch.tensor([1], dtype=torch.float)
num_dim  = 2 # number of dimensions
num_modes = means.shape[0]
scale_factor = 2
covs = scale_factor * np.ones((num_modes, num_dim))

gauss = nf.distributions.GaussianMixture(n_modes = 2, dim = 2, loc=means, scale = covs, weights=weights, trainable=False)
#plot_gm_dist(gauss, 100000)


#%%

###################### MODEL STARTS HERE ######################
# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

K = 12
flows = []
for i in range(K):
    flows += [nf.flows.Planar((2,))]

print(flows)

#%%

#q0 = nf.distributions.DiagGaussian(2)
#plot_gm_dist(q0, 50000)

base = grid_gaussian
target = three_gaussians
plot_gm_dist(q0, 10000)
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
max_iter = 10000
num_samples = 200
anneal_iter = 10000
annealing = True # what is annealing?
show_iter = 500
plotting = True

#%%

loss_hist, nfm = train_flow_model(nfm, max_iter, num_samples, anneal_iter, annealing, show_iter, plotting)

#%%

plot_learned_model(nfm, 10000)


#%% TODO: define an analysis class for the flow network
# 1. plot each flow independently
# 2. plot the sequence of flows starting from the base to the target

num_samples = 10000
base = grid_gaussian
xy_all = flow_layer_pass_through(base, nfm, num_samples, K)

plot_flow_layers(xy_all) # use a base distribution as input

#%%

# pass a grid_gaussian through each flow layer
# plot the output of each flow layer

def grid_flow_pass_through(grid_gauss, nfm, num_samples, K):
    xy_all = []
    for i in range(K):
        xy = flow_layer_pass_through(grid_gauss, nfm, num_samples, i)
        xy_all.append(xy)
    return xy_all


means, weights =  grid_gaussian_params(4)
num_dim  = 2 # number of dimensions
num_modes = means.shape[0]
scale_factor = 0.1
covs = scale_factor * np.ones((num_modes, num_dim))

grid_gaussian_plot = nf.distributions.GaussianMixture(n_modes = num_modes, dim = num_dim, loc=means, scale = covs, weights=weights, trainable=False)
plot_gm_dist(grid_gaussian_plot, 100000)

num_samples = 10000
flow_grid = grid_flow_pass_through(grid_gaussian_plot, nfm, num_samples, K)
plot_flow_layers(xy_all) 


# %%

# %%
