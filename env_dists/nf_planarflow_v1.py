#%% planar flow example - checking the inner layers

# Import required packages
import torch
import numpy as np
import normflows as nf

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
        
        plt.figure(figsize=(10, 10))
        plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (grid_size, grid_size), range=[[-3, 3], [-3, 3]])
        plt.show()

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

num_samples = 1000
layer_n = 0
z_test  = [[nfm.flows[layer_n].forward(e)[0].detach().numpy(), 
            nfm.flows[layer_n].forward(e)[1].detach().numpy()] 
            for e in np.random.normal(0,10,num_samples)]

xy = np.zeros((len(z_test),2))
zz = np.zeros(len(z_test))

for i,e in enumerate(z_test):
    xy[i,:] = e[0][0]
    zz[i] = e[1]

#%%
# plot the 2d hist of xy
plt.figure(figsize=(10, 10))
plt.hist2d(xy[:,0], xy[:,1], (grid_size, grid_size), range=[[-3, 3], [-3, 3]])
plt.show()

#%%
# plot the 2d scatter of xy
plt.figure(figsize=(10, 10))
plt.scatter(xy[:,0], xy[:,1], c=zz)
plt.show()

# %%

# function that plots the 2d hist of xy for a given layer
##### why is this giving me lines?

def plot_hist_layer(layer_n, num_samples, grid_size):
    z_test  = [[nfm.flows[layer_n].forward(e)[0].detach().numpy(), 
                nfm.flows[layer_n].forward(e)[1].detach().numpy()] 
                for e in np.random.normal(0,10,num_samples)]

    xy = np.zeros((len(z_test),2))
    zz = np.zeros(len(z_test))

    for i,e in enumerate(z_test):
        xy[i,:] = e[0][0]
        zz[i] = e[1]

    plt.figure(figsize=(10, 10))
    plt.hist2d(xy[:,0], xy[:,1], (grid_size, grid_size), range=[[-3, 3], [-3, 3]])
    plt.show()

num_samples = 1000
for layer in range(K):
    plot_hist_layer(layer, num_samples, grid_size=200)




