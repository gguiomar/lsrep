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

sim_data_opt_dm = run_information_bottleneck_DM(rho_opt, pi_z_init_opt, n_steps, beta)
#sim_data_conf_dm = run_information_bottleneck_DM(rho_conf, pi_z_init_conf, n_steps, beta)

#sim_data_opt_gg = run_information_bottleneck_GG(rho_opt.T, pi_z_init_conf, n_steps, beta)

#sim_data_opt = run_information_bottleneck(rho_opt.T, pi_z_init_conf.T, n_steps, beta)


# rho_o = sim_data_opt_dm['rho']
# rho_c = sim_data_conf_dm['rho']
# pi_z_o = sim_data_opt_dm['pi_z']
# pi_z_c = sim_data_conf_dm['pi_z']

#%%
#plot_sim_metrics(sim_data_opt_dm, [25,17])
plot_sim_metrics(sim_data_opt_gg, [25,17])

#%%

plot_rho_pi_comparison(sim_data_opt_dm, sim_data_conf_dm)

# %% recoding the ib simulation - making indexes notationally correct

rho = np.array([[.9, .1], [.9, .1], [.1, .9], [.1, .9]]).T
pi_z_init= np.array([[0.99, 0.01], [0.01, 0.99]])

n_states = 4
n_latent = 2
n_actions = 2

pi_z = pi_z_init
pi_behave = np.ones((n_actions, n_states))/n_actions # behavioral policy induced through latent representation
pi_s = np.array([[0.99, 0.01], [0.99, 0.01], [0.01, 0.99], [0.01, 0.99]]).T # target policy

p_z = np.ones((1, n_latent))/n_latent
p_s = np.ones((1, n_states))/n_states
p_sz = np.ones((n_states, n_latent))/n_latent

error_behave = []
pi_z_h = []

for t in range(1):
    for s in range(n_states):
        for z in range(n_latent):
            rho[z,s] = p_z[0, z]*np.exp(-beta*kl(pi_s[:,s], pi_z[:,z])) #check this implementation
        rho[:,s] /= (rho[:,s]).sum()

    print('rho', rho.shape)
    # compute marginal latents
    p_z = (p_s * rho).sum(1).reshape(1, n_latent)

    print('p_z', p_z.shape)
    # compute environment state
    p_sz = (rho * p_s)
    print('p_sz0', p_sz.shape)
    p_sz /= (p_sz.sum(1)).reshape(n_latent,1)

    print('p_sz', p_sz.shape)
    # compute latent policy
    pi_z = np.dot(rho, pi_s.T)
    pi_z_h.append(pi_z)

    print('pi_z', pi_z.shape)
    # compute behavioral policy
    pi_behave = np.dot(rho.T, pi_z).T

    print('pi_behave', pi_behave.shape)
    # record KL between target policy and behavioral policy
    error_behave.append(error2(p_s, pi_behave, pi_s))

# %%
