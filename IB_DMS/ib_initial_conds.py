#ib_initial_conds.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams["font.size"] = "20"

# create list of initial conditions
# simulate multiple runs

# PI_Z INITS

def get_inits():
    
    n_actions = 2
    n_latent = 2
    
    pi_z_inits_ds = []
    rho_inits = []
    rho_opt = np.array([[.9, .1], [.9, .1], [.1, .9], [.1, .9]]).T
    rho_conf = np.array([[.1, .9], [.1, .9], [.9, .1], [.9, .1]]).T

    pi_z = np.zeros((n_actions, n_latent))
    pi_z[0,0] = 0.01
    pi_z[0,1] = 0.01
    pi_z[1,0] = 0.01
    pi_z[1,1] = 0.01
    pi_z = pi_z / pi_z.sum(axis=0)
    pi_z_inits_ds.append(pi_z)
    rho_inits.append(rho_opt)

    pi_z = np.zeros((n_actions, n_latent))
    pi_z[0,0] = 0.02
    pi_z[0,1] = 0.01
    pi_z[1,0] = 0.01
    pi_z[1,1] = 0.01
    pi_z = pi_z / pi_z.sum(axis=0)
    pi_z_inits_ds.append(pi_z)
    rho_inits.append(rho_opt)

    pi_z = np.zeros((n_actions, n_latent))
    pi_z[0,0] = 0.01
    pi_z[0,1] = 0.02
    pi_z[1,0] = 0.01
    pi_z[1,1] = 0.01
    pi_z = pi_z / pi_z.sum(axis=0)
    pi_z_inits_ds.append(pi_z)
    rho_inits.append(rho_conf)

    pi_z = np.zeros((n_actions, n_latent))
    pi_z[0,0] = 0.01
    pi_z[0,1] = 0.01
    pi_z[1,0] = 0.02
    pi_z[1,1] = 0.01
    pi_z = pi_z / pi_z.sum(axis=0)
    pi_z_inits_ds.append(pi_z)
    rho_inits.append(rho_conf)

    pi_z = np.zeros((n_actions, n_latent))
    pi_z[0,0] = 0.01
    pi_z[0,1] = 0.01
    pi_z[1,0] = 0.01
    pi_z[1,1] = 0.02
    pi_z = pi_z / pi_z.sum(axis=0)
    pi_z_inits_ds.append(pi_z)
    rho_inits.append(rho_opt)

    pi_z = np.zeros((n_actions, n_latent))
    pi_z[0,0] = 0.02
    pi_z[0,1] = 0.02
    pi_z[1,0] = 0.01
    pi_z[1,1] = 0.01
    pi_z = pi_z / pi_z.sum(axis=0)
    pi_z_inits_ds.append(pi_z)
    rho_inits.append(rho_conf)

    pi_z = np.zeros((n_actions, n_latent))
    pi_z[0,0] = 0.01
    pi_z[0,1] = 0.01
    pi_z[1,0] = 0.02
    pi_z[1,1] = 0.02
    pi_z = pi_z / pi_z.sum(axis=0)
    pi_z_inits_ds.append(pi_z)
    rho_inits.append(rho_opt)

    pi_z = np.zeros((n_actions, n_latent))
    pi_z[0,0] = 0.02
    pi_z[0,1] = 0.01
    pi_z[1,0] = 0.01
    pi_z[1,1] = 0.02
    pi_z = pi_z / pi_z.sum(axis=0)
    pi_z_inits_ds.append(pi_z)
    rho_inits.append(rho_opt)

    pi_z = np.zeros((n_actions, n_latent))
    pi_z[0,0] = 0.01
    pi_z[0,1] = 0.02
    pi_z[1,0] = 0.02
    pi_z[1,1] = 0.01
    pi_z = pi_z / pi_z.sum(axis=0)
    pi_z_inits_ds.append(pi_z)
    rho_inits.append(rho_conf)
    
    return pi_z_inits_ds, rho_inits


def plot_inits(inits):
    
    fig = plt.figure(figsize = [7,7])
    gs = GridSpec(4, 3)
    ax = []
    vmin = 0
    vmax = 1
    cmap = 'Greys'
    for i,e in enumerate(inits):
        ax.append([])
        ax[i] = fig.add_subplot(gs[i])
        ax[i].imshow(e, cmap, vmin = vmin, vmax = vmax)
        ax[i].tick_params(labelbottom=False, labelleft=False, bottom = False, left = False)
    plt.show()
    
## RHO INITS