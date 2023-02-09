#%% planar flow example - checking the inner layers

# Import required packages
import torch
import numpy as np
import normflows as nf
from sklearn import cluster, datasets, mixture
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
from matplotlib import pyplot as plt
from tqdm import tqdm

#%%
#torch.manual_seed(0)

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

K = 16
flows = []
for i in range(K):
    flows += [nf.flows.Planar((2,))]
target = nf.distributions.TwoModes(2, 0.1)

q0 = nf.distributions.DiagGaussian(2) # what's the dimensionality of these samples?
nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=target)
nfm.to(device)

#%%

# Plot target distribution
grid_size = 200
xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
z = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
log_prob = target.log_prob(z.to(device)).to('cpu').view(*xx.shape)
prob = torch.exp(log_prob)

plt.figure(figsize=(10, 10))
plt.pcolormesh(xx, yy, prob)
plt.show()

# Plot initial flow distribution
z, _ = nfm.sample(num_samples=2 ** 20)
z_np = z.to('cpu').data.numpy()
plt.figure(figsize=(10, 10))
plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (grid_size, grid_size), range=[[-3, 3], [-3, 3]])
plt.show()

#%% 

# Train model
max_iter = 20000
num_samples = 2 * 20
anneal_iter = 10000
annealing = True
show_iter = 2000


def train_flow_model(nfm, max_iter, num_samples, anneal_iter, annealing, show_iter):

    loss_hist = np.array([])
    optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-3, weight_decay=1e-4)
    for it in tqdm(range(max_iter)):
        optimizer.zero_grad()
        if annealing:
            loss = nfm.reverse_kld(num_samples, beta=np.min([1., 0.01 + it / anneal_iter]))
        else:
            loss = nfm.reverse_kld(num_samples)
        loss.backward()
        optimizer.step()
        
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        
        # Plot learned distribution
        if (it + 1) % show_iter == 0:
            torch.cuda.manual_seed(0)
            z, _ = nfm.sample(num_samples=2 ** 20)
            z_np = z.to('cpu').data.numpy()

    return loss_hist, z_np, z

loss_hist, z_np, z = train_flow_model(nfm, max_iter, num_samples, anneal_iter, annealing, show_iter)

#%%

# plt.figure(figsize=(10, 10))
# plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (grid_size, grid_size), range=[[-3, 3], [-3, 3]])
# plt.show()

plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()

#%%

# Plot learned distribution
z, _ = nfm.sample(num_samples=2 ** 20)
z_np = z.to('cpu').data.numpy()
plt.figure(figsize=(10, 10))
plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (grid_size, grid_size), range=[[-3, 3], [-3, 3]])
plt.show()
# %%

# %%

# create a function that passes a 2d gaussian dist through the flow layers

def flow_layer_forward(layer_n, num_samples, grid_size):
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
    fig, axs = plt.subplots(4, 4, figsize=(15,15))
    for i in range(4):  
        for j in range(4):
            axs[i,j].hist2d(xy_all[i*4+j,:,0], xy_all[i*4+j,:,1], (grid_size_plot, grid_size_plot), range=[[-3, 3], [-3, 3]])
            axs[i,j].set_title('Layer ' + str(i*4+j))
    plt.show()

def flow_layer_previous(num_samples, n_layers):
    q0 = nf.distributions.DiagGaussian(2)
    z, _ = q0.forward(num_samples= num_samples)

    # run the q0 samples through the flow layers and plot the results
    xy = np.zeros((n_layers, len(z), 2))
    for k in range(n_layers):
        if k == 0:
            z, _ = q0.forward(num_samples = num_samples)
        else:
            # change xy to tensor format
            z = torch.from_numpy(xy[k-1,:,:]).float()

        for i,e in enumerate(z):
            xy[k, i,:] = nfm.flows[k].forward(e)[0].detach().numpy()

    return xy
#%%

# save all the layer outputs
num_samples = 1000
xy_all = np.zeros((K, num_samples, 2))
for i in range(K):
    xy_all[i,:,:] = flow_layer_forward(i, num_samples, grid_size)
plot_flow_layers(xy_all)

# %%
num_samples = 10000
xy_all = np.zeros((K, num_samples, 2))
xy_all = flow_layer_previous(num_samples, K)
plot_flow_layers(xy_all)

#%% ----------------------------------------------

# define the number of samples to be drawn
n_samples = 100000

# define the mean points for each of the systhetic cluster centers
# spread them out in a 10x10 grid

t_means = []
for i in range(-5, 5):
    for j in range(-5, 5):
        t_means.append([i, j])

#t_means = [[0, 0], [1, 1], [-1, 1], [-1, -1]]

# for each cluster center, create an identity covariance matrix

t_covs = []
for s in range(len(t_means)):
    t_covs.append(0.01 * np.identity(2))

# t_covs = []
# for s in range(len(t_means)):
#   t_covs.append(make_spd_matrix(2))

X = []
for mean, cov in zip(t_means,t_covs):
  x = np.random.multivariate_normal(mean, cov, n_samples)
  X += list(x)
  
X = np.array(X)

# plot the histogram of X

grid_size_plot = 100
plt.figure(figsize=(10, 10))
plt.hist2d(X[:, 0].flatten(), X[:, 1].flatten(), (grid_size_plot, grid_size_plot), range=[[-3, 3], [-3, 3]])
plt.show()

# define a function that generates a grid of gaussian distributions
#%%

def make_grid_gaussian_mixture(grid_size, n_samples_per_gauss):

    t_means = []
    for i in range(-int(grid_size/2), int(grid_size/2)):
        for j in range(-int(grid_size/2), int(grid_size/2)):
            t_means.append([i, j])

    t_covs = []
    for s in range(len(t_means)):
        t_covs.append(0.01 * np.identity(2))

    X = []
    for mean, cov in zip(t_means,t_covs):
      x = np.random.multivariate_normal(mean, cov, n_samples_per_gauss)
      X += list(x)
      
    xy = np.array(X)

    return xy


grid_size_gauss = 10
n_samples_per_gauss = 100
xy_grid = make_grid_gaussian_mixture(grid_size_gauss, n_samples_per_gauss)
xy_grid.shape
# plot the histogram of xy_grid

grid_size_plot = 100
plt.figure(figsize=(10, 10))
plt.hist2d(xy_grid[:, 0].flatten(), xy_grid[:, 1].flatten(), (grid_size_plot, grid_size_plot), range=[[-3, 3], [-3, 3]])
plt.show()


# %%

# pass the grid of gaussian distributions through the flow layers

#def flow_layer_forward_dist(nfm, n_layers, num_samples, dist):
    
z = torch.from_numpy(xy_grid).float()
xy = np.zeros((K, len(z),2))
print(z.shape)

for k in range(K):
    print('layer: ', k)
    for i,e in enumerate(z):
        xy[k,i,:] = nfm.flows[k].forward(e)[0].detach().numpy()

#    return xy


xy_grid_pass = flow_layer_forward_dist(nfm, K, 10000, xy_grid)

plot_flow_layers(xy_grid_pass)

    # %%
