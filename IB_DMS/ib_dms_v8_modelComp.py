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

#%%

n_state = 4
n_latent = 2
n_actions = 2
n_steps = 10
beta = 100

# opt
rho = np.array([[.9, .1], [.9, .1], [.1, .9], [.1, .9]]).T
pi_z_init= np.array([[0.99, 0.01], [0.01, 0.99]])
sim_data = run_information_bottleneck(rho, pi_z_init, 1, beta)
plot_sim_metrics(sim_data, [25,17])

#%%

# conf
rho = np.array([[.1, .9], [.1, .9], [.9, .1], [.9, .1]]).T
pi_z_init = np.array([[0.01, 0.99],[0.99, 0.01]])
sim_data = run_information_bottleneck(rho, pi_z_init, 1, beta)
plot_sim_metrics(sim_data, [25,17])


# %%
inits = get_inits()
sim_data = []
for e,d in zip(inits[1], inits[0]):
    sim_data.append(run_information_bottleneck(e,d,1,beta))
    
for e in sim_data:
    plot_sim_metrics(e, [25,17])

# %%
