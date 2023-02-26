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

#%%
# FLOW LAYERS FUNC 

def train_flow_model(nfm, max_iter, num_samples, anneal_iter, 
                    annealing, show_iter, plotting, save_fig, make_gif):

    #get working directory
    cwd = os.getcwd()
    path = cwd + '/plots/' + time.strftime("%Y%m%d-%H%M%S")
    if save_fig:
        os.mkdir(path)

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
        
        # saving simulation figures
        image_names = []
        it_order = []
        if (it + 1) % show_iter == 0:
            it_order.append(it + 1)
            fig = plt.figure(figsize=(10, 10))
            torch.cuda.manual_seed(0)
            z, _ = nfm.sample(num_samples=2 ** 20)
            z_np = z.to('cpu').data.numpy()
            plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (grid_size, grid_size), range=[[-3,3], [-3, 3]])
            if plotting:
                plt.show()
            if save_fig: # save figure in current directory with current timestamp directory and iteration on filename
                image_names.append(path + '/' + str(it + 1) + '_flow_train' + '.png')
                fig.savefig(image_names[-1])
        
    if make_gif: # make gif from saved images
        images = []
        print(image_names)
        for filename in image_names:
            images.append(imageio.imread(filename))
        imageio.mimsave(path + '/flow_train.gif', images, duration=0.2)

    return loss_hist, nfm, path


# list all files in folder

def list_files(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".png"):
            files.append(os.path.join(path, file))
    return files


def make_gif_from_folder(folder_name, gif_name):

    images = []

    #list files
    file_list = sorted(os.listdir(folder_name))

    # split file name by underscore
    file_list_split = [np.asarray(file.split('_')) for file in file_list]

    # sort by first element magnitude
    file_list_index = np.asarray([np.asarray([i, int(e[0])]) for i,e in enumerate(file_list_split[1:-1])])

    # sort by second element magnitude
    file_list_index = sorted(file_list_index, key=lambda x: x[1])

    # get file names in order

    file_list_ordered = [file_list[i[0]+1] for i in file_list_index]

    for file_name in file_list_ordered:
        if file_name.endswith('.png'):
            file_path = os.path.join(folder_name, file_name)
            images.append(imageio.imread(file_path))
            #imageio.mimsave(gif_name, images)

    imageio.mimsave(os.path.join(folder_name, gif_name), images)

def flow_layer_forward(nfm, layer_n, num_samples):

    q0 = nf.distributions.DiagGaussian(2)
    z, _ = q0.forward(num_samples= num_samples)

    # run the q0 samples through the flow layers and plot the results
    xy = np.zeros((len(z),2))
    for i,e in enumerate(z):
        xy[i,:] = nfm.flows[layer_n].forward(e)[0].detach().numpy()

    return xy

# plot all the layers as a 4x4 grid
def plot_flow_layers(xy_all, range_v):

    grid_size_plot = 100
    #range_v = [[-3,3], [-3, 3]]

    # number of plots
    n_plots = xy_all.shape[0]
    # if the number of plots is not a square number, add extra plots
    n_rows = int(np.ceil(np.sqrt(n_plots)))
    n_cols = int(np.ceil(n_plots/n_rows))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15,15))
    for i in range(n_rows):  
        for j in range(n_cols):
            if i*n_cols + j < n_plots:
                axs[i,j].hist2d(xy_all[i*n_cols + j,:,0].flatten(), xy_all[i*n_cols + j,:,1].flatten(), (grid_size_plot, grid_size_plot), range=range_v)
                axs[i,j].set_title('Layer ' + str(i*n_cols + j))
                axs[i,j].set_xlim(range_v[0])
                axs[i,j].set_ylim(range_v[1])
            else:
                axs[i,j].axis('off')
    plt.show()

def plot_flow_layers_save(xy_all, range_v, filename, path):

    grid_size_plot = 100
    #range_v = [[-3,3], [-3, 3]]

    # number of plots
    n_plots = xy_all.shape[0]
    # if the number of plots is not a square number, add extra plots
    n_rows = int(np.ceil(np.sqrt(n_plots)))
    n_cols = int(np.ceil(n_plots/n_rows))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15,15))
    for i in range(n_rows):  
        for j in range(n_cols):
            if i*n_cols + j < n_plots:
                axs[i,j].hist2d(xy_all[i*n_cols + j,:,0].flatten(), xy_all[i*n_cols + j,:,1].flatten(), (grid_size_plot, grid_size_plot), range=range_v)
                axs[i,j].set_title('Layer ' + str(i*n_cols + j))
                axs[i,j].set_xlim(range_v[0])
                axs[i,j].set_ylim(range_v[1])
            else:
                axs[i,j].axis('off')
    if save_fig: # save figure with file name
        fig.savefig(path + '/' + filename + '.png')
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

def grid_flow_pass_through(grid_gauss, nfm, num_samples, K):
    
    z, _ = grid_gauss.forward(num_samples = num_samples)

    xy_all = np.zeros((K, num_samples, 2))
    for k in range(K):
        for i,e in enumerate(z):
            xy_all[k,i,:] = nfm.flows[k].forward(e)[0].detach().numpy()
    return xy_all

# PLOT DISTRIBUTIONS

def plot_gm_dist(p, num_samples):

    grid_size = 100
    z, _ = p.forward(num_samples)
    z = z.detach().numpy()
    plt.hist2d(z[:, 0].flatten(), z[:, 1].flatten(), (grid_size, grid_size), range=[[-3,3], [-3, 3]])
    plt.show()

def plot_twomoon_dist(p, num_samples):

    grid_size = 100
    z = p.sample(num_samples)
    z = z.detach().numpy()
    plt.hist2d(z[:, 0].flatten(), z[:, 1].flatten(), (grid_size, grid_size), range=[[-3,3], [-3, 3]])
    plt.show()

def plot_learned_model(nfm, num_samples):
    grid_size = 100
    z, _ = nfm.sample(num_samples=2**20)
    z_np = z.to('cpu').data.numpy()
    plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (grid_size, grid_size), range=[[-3, 3], [-3, 3]])
    plt.show()

# GENERATE DISTRIBUTIONS

def grid_gaussian_params(N):

    means = []
    for i in range(-N, N):
        for j in range(-N, N):
            means.append([i, j])

    means = torch.tensor(means, dtype=torch.float)
    weights = torch.tensor([1/len(means)] * len(means), dtype=torch.float)

    return means, weights

def generate_psa_dist(scale_factor):

    means = torch.tensor([[-2, -2], [2, 2], [0, 2], [-0,-2]], dtype=torch.float)
    weights = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float)
    num_dim  = 2 # number of dimensions
    num_modes = means.shape[0]
    covs = scale_factor * np.ones((num_modes, num_dim))
    psa_dist = nf.distributions.GaussianMixture(n_modes = num_modes, dim = num_dim, loc=means, scale = covs, weights=weights, trainable=False)

    return psa_dist

def generate_grid_dist(n_grid, scale_factor):

    means, weights =  grid_gaussian_params(n_grid)
    num_dim  = 2 # number of dimensions
    num_modes = means.shape[0]
    covs = scale_factor * np.ones((num_modes, num_dim))
    grid_dist = nf.distributions.GaussianMixture(n_modes = num_modes, dim = num_dim, loc=means, scale = covs, weights=weights, trainable=False)

    return grid_dist


def generate_twomoon_dist():

    two_moon = nf.distributions.TwoMoons()

    return two_moon

def generate_2gauss_dist(scale_factor):

    means = torch.tensor([[-1, 0], [1, 0]], dtype=torch.float)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float)
    num_dim  = 2 # number of dimensions
    num_modes = means.shape[0]
    scale_factor = 0.3
    covs = scale_factor * np.ones((num_modes, num_dim))
    two_gauss = nf.distributions.GaussianMixture(n_modes = num_modes, dim = num_dim, loc=means, scale = covs, weights=weights, trainable=False)

    return two_gauss


# generate base, target distributions

def generate_base_and_target(base_name, target_name, n_grid = 3, scale_factor = 0.3):

    if base_name == 'grid':
        base = generate_grid_dist(n_grid, scale_factor)
    elif base_name == 'psa':
        base = generate_psa_dist(scale_factor)
    elif base_name == 'twomoon':
        base = generate_twomoon_dist()
    elif base_name == '2gauss':
        base = generate_2gauss_dist(scale_factor)
    elif base_name == 'gauss':
        base = nf.distributions.DiagGaussian(2)

    if target_name == 'grid':
        target = generate_grid_dist(n_grid, scale_factor)
    elif target_name == 'psa':
        target = generate_psa_dist(scale_factor)
    elif target_name == 'twomoon':
        target = generate_twomoon_dist()
    elif target_name == '2gauss':
        target = generate_2gauss_dist(scale_factor)

    return base, target

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

base, target = generate_base_and_target('gauss', 'psa', n_grid = 4, scale_factor = 0.3)
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
max_iter = 10000
num_samples = 1000
anneal_iter = 1000
annealing = True # what is annealing?
show_iter = 100
plotting = False
make_gif = True
save_fig = True

#%%

loss_hist, nfm, path = train_flow_model(nfm, max_iter, num_samples, anneal_iter, 
                                  annealing, show_iter, plotting, save_fig, make_gif)

#%%

plot_learned_model(nfm, 10000)


#%% 

num_samples = 10000
base = nf.distributions.DiagGaussian(2)
xy_all = flow_layer_pass_through(base, nfm, num_samples, K)
path = 'plots/20230220-134258/'
range_v = [[-3, 3], [-3, 3]]
filename = 'pass_through_gaussian'

plot_flow_layers_save(xy_all, range_v, filename, path)
#%%

num_samples = 10000
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

plot_flow_layers_save(xy_all, range_v, filename, path)


#%%

# pass a grid_gaussian through each flow layer
# plot the output of each flow layer

means, weights =  grid_gaussian_params(4)
num_dim  = 2 # number of dimensions
num_modes = means.shape[0]
scale_factor = 0.1
covs = scale_factor * np.ones((num_modes, num_dim))

grid_gaussian_plot = nf.distributions.GaussianMixture(n_modes = num_modes, dim = num_dim, loc=means, scale = covs, weights=weights, trainable=False)
#plot_gm_dist(grid_gaussian_plot, 100000)

num_samples = 1000
flow_grid = grid_flow_pass_through(grid_gaussian_plot, nfm, num_samples, K)
rs = 6
range_v = [[-rs,rs],[-rs,rs]]
plot_flow_layers(flow_grid, range_v) # use a base distribution as input


# %%

# make a gif from the images in the folder ordered by last digits in the file name

#%% make gif

folder_name = 'plots/20230220-134258/'
gif_name = 'gif_test2.gif'

make_gif_from_folder(folder_name, gif_name)


# %%
