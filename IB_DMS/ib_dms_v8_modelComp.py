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
from ib_initial_conds import *

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
plot_sim_metrics(sim_data_opt, [25,17])

#%%

plot_rho_pi_comparison(sim_data_opt, sim_data_conf)


# %%
