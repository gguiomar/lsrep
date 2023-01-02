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

n_states = 4
n_latent = 2
n_actions = 2
n_steps = 10
beta = 100

# %%
inits = get_inits()
sim_data = []
for e in inits:
    sim_data.append(run_information_bottleneck(e,n_steps,beta))
    
for e in sim_data:
    plot_sim_metrics(e, [25,17])
    
# %%
