#!/usr/bin/python

# My take-away intuitions:
# 1. Need noise in latent policy initialization otherwise stuck on nullcline (i.e. algorithm can't "decide" which way to go)
# 2. Increase beta for lower error and faster convergence

#%% inits

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams["font.size"] = "20"

# simulation and plotting tools
from information_bottleneck_tools import *

n_state = 4
n_latent = 2
n_actions = 2

n_steps = 10
beta = 100

rho_opt = np.array([[.9, .1], [.9, .1], [.1, .9], [.1, .9]])
rho_conf = np.array([[.1, .9], [.1, .9], [.9, .1], [.9, .1]])
pi_z_init_opt = np.array([[0.99, 0.01], [0.01, 0.99]])
pi_z_init_conf = np.array([[0.01, 0.99],[0.99, 0.01]])
sim_data_opt = run_information_bottleneck_DM(rho_opt, pi_z_init_opt, n_steps, beta)
sim_data_conf = run_information_bottleneck_DM(rho_conf, pi_z_init_conf, n_steps, beta)


rho_o = sim_data_opt['rho']
rho_c = sim_data_conf['rho']
pi_z_o = sim_data_opt['pi_z']
pi_z_c = sim_data_conf['pi_z']

#%%

fig, axes = plt.subplots(2,4, figsize = (20,17))
vmin = 0; vmax = 1.; cmap = 'Greys'

axes[0,0].plot(range(n_steps+1), sim_data_opt['error_behave'], 'k')
axes[0,0].set(ylabel = 'KL', xlabel = 'iterations', title = 'Optimal IC', yticks = [0,1], aspect=6)
axes[0,0].spines['right'].set_visible(False)
axes[0,0].spines['top'].set_visible(False)
axes[0,1].imshow(sim_data_opt['pi_z'], vmin=vmin, vmax=vmax, cmap=cmap)
axes[0,1].set(ylabel = 'action', xlabel = 'latent state (z)', xticks = [0,1], yticks = [0,1], title = 'learned $\pi_z$')
axes[0,2].imshow(sim_data_opt['rho'], vmin=vmin, vmax=vmax, cmap=cmap)
axes[0,2].set(xlabel = 'action', ylabel = 'observed state (s)', xticks = [0,1], yticks = [0,1,2,3], title = r'$\rho(z|s)$', aspect = 0.5)
axes[0,3].imshow(np.dot(rho_o,pi_z_o), vmin=vmin, vmax=vmax, cmap=cmap)
axes[0,3].set(xlabel = 'action', ylabel = 'observed state (s)', xticks = [0,1], yticks = [0,1,2,3], title = 'reconstructed $\pi_s$', aspect = 0.5)

axes[1,0].plot(range(n_steps+1), sim_data_conf['error_behave'], 'k')
axes[1,0].set(ylabel = 'KL', xlabel = 'iterations', title = 'Confused IC', yticks = [0,1], aspect=6)
axes[1,0].spines['top'].set_visible(False)
axes[1,0].spines['right'].set_visible(False)
axes[1,1].imshow(sim_data_conf['pi_z'], vmin=vmin, vmax=vmax, cmap=cmap)
axes[1,1].set(xlabel = 'latent state (z)', ylabel = 'action', xticks = [0,1], yticks = [0,1], title = 'learned $\pi_z$') 
axes[1,2].imshow(sim_data_conf['rho'], vmin=vmin, vmax=vmax, cmap=cmap)
axes[1,2].set(xlabel = 'action', ylabel = 'observed state (s)', xticks = [0,1], yticks = [0,1,2,3], title = r'$\rho(z|s)$', aspect = 0.5)
axes[1,3].imshow(np.dot(rho_c,pi_z_c), vmin=vmin, vmax=vmax, cmap=cmap)
axes[1,3].set(xlabel = 'action', ylabel = 'observed state (s)', xticks = [0,1], yticks = [0,1,2,3], title = 'reconstructed $\pi_s$', aspect = 0.5)

plt.show()

#%%

plot_rho_pi_comparison(sim_data_opt, sim_data_conf)

# %% checking ICs of comparison

plt.imshow(sim_data_opt['pi_z_h'][0], 'Greys')
plt.show()
plt.imshow(sim_data_conf['pi_z_h'][0], 'Greys')
plt.show()


# %% defining ICs 

# create list of initial conditions
# simulate multiple runs
pi_z_inits_ds = []

pi_z = np.zeros((n_actions, n_latent))
pi_z[0,0] = 0.01
pi_z[0,1] = 0.01
pi_z[1,0] = 0.01
pi_z[1,1] = 0.01
pi_z = pi_z / pi_z.sum(axis=0)
pi_z_inits_ds.append(pi_z)

pi_z = np.zeros((n_actions, n_latent))
pi_z[0,0] = 0.02
pi_z[0,1] = 0.01
pi_z[1,0] = 0.01
pi_z[1,1] = 0.01
pi_z = pi_z / pi_z.sum(axis=0)
pi_z_inits_ds.append(pi_z)

pi_z = np.zeros((n_actions, n_latent))
pi_z[0,0] = 0.01
pi_z[0,1] = 0.02
pi_z[1,0] = 0.01
pi_z[1,1] = 0.01
pi_z = pi_z / pi_z.sum(axis=0)
pi_z_inits_ds.append(pi_z)

pi_z = np.zeros((n_actions, n_latent))
pi_z[0,0] = 0.01
pi_z[0,1] = 0.01
pi_z[1,0] = 0.02
pi_z[1,1] = 0.01
pi_z = pi_z / pi_z.sum(axis=0)
pi_z_inits_ds.append(pi_z)

pi_z = np.zeros((n_actions, n_latent))
pi_z[0,0] = 0.01
pi_z[0,1] = 0.01
pi_z[1,0] = 0.01
pi_z[1,1] = 0.02
pi_z = pi_z / pi_z.sum(axis=0)
pi_z_inits_ds.append(pi_z)

pi_z = np.zeros((n_actions, n_latent))
pi_z[0,0] = 0.02
pi_z[0,1] = 0.02
pi_z[1,0] = 0.01
pi_z[1,1] = 0.01
pi_z = pi_z / pi_z.sum(axis=0)
pi_z_inits_ds.append(pi_z)

pi_z = np.zeros((n_actions, n_latent))
pi_z[0,0] = 0.01
pi_z[0,1] = 0.01
pi_z[1,0] = 0.02
pi_z[1,1] = 0.02
pi_z = pi_z / pi_z.sum(axis=0)
pi_z_inits_ds.append(pi_z)

pi_z = np.zeros((n_actions, n_latent))
pi_z[0,0] = 0.02
pi_z[0,1] = 0.01
pi_z[1,0] = 0.01
pi_z[1,1] = 0.02
pi_z = pi_z / pi_z.sum(axis=0)
pi_z_inits_ds.append(pi_z)

pi_z = np.zeros((n_actions, n_latent))
pi_z[0,0] = 0.01
pi_z[0,1] = 0.02
pi_z[1,0] = 0.02
pi_z[1,1] = 0.01
pi_z = pi_z / pi_z.sum(axis=0)
pi_z_inits_ds.append(pi_z)

fig = plt.figure(figsize = [7,7])
gs = GridSpec(4, 3)
ax = []
for i,e in enumerate(pi_z_inits_ds):
    ax.append([])
    ax[i] = fig.add_subplot(gs[i])
    ax[i].imshow(e, 'Blues', vmin = vmin, vmax = vmax)
    ax[i].tick_params(labelbottom=False, labelleft=False, bottom = False, left = False)
plt.show()


# %%
