#%% defining all the utility functions

import torch
import numpy as np
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import tqdm
import copy

import imageio
import os
import time

def train_flow_model(nfm, max_iter, num_samples, anneal_iter, 
                    annealing, show_iter, plotting, save_fig):

    #get working directory
    cwd = os.getcwd()
    path = cwd + '/plots/' + time.strftime("%Y%m%d-%H%M%S")
    if save_fig:
        os.mkdir(path)

    loss_hist = np.array([])
    optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-3, weight_decay=1e-4)
    grid_size = 100

    nfm_h = []

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
        if (it + 1) % show_iter == 0:
            # copy the flows object
            nfm_h.append(copy.deepcopy(nfm))

            # sample the model at this stage of training
            fig = plt.figure(figsize=(10, 10))
            torch.cuda.manual_seed(0)
            z, _ = nfm.sample(num_samples=2 ** 20)
            z_np = z.to('cpu').data.numpy()
            plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (grid_size, grid_size), range=[[-3,3], [-3, 3]])

            if plotting:
                plt.show()
            if save_fig: 
                fig.savefig(path + '/' + str(it+1) + '_training_flow' + '.png')

    return loss_hist, nfm, path, nfm_h


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
    if path: # save figure with file name
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