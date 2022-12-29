#ib_initial_conds.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams["font.size"] = "20"

# create list of initial conditions
# simulate multiple runs

# PI_Z INITS

def get_pi_z_inits(n_actions, n_latent):
    
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
    
    return pi_z_inits_ds


def plot_pi_zinits(pi_z_inits_ds):
    
    fig = plt.figure(figsize = [7,7])
    gs = GridSpec(4, 3)
    ax = []
    for i,e in enumerate(pi_z_inits_ds):
        ax.append([])
        ax[i] = fig.add_subplot(gs[i])
        ax[i].imshow(e, cmap, vmin = vmin, vmax = vmax)
        ax[i].tick_params(labelbottom=False, labelleft=False, bottom = False, left = False)
    plt.show()
    
## RHO INITS