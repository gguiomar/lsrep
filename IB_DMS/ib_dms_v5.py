import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams.update({'font.size': 13})


# %%
# auxiliary functions

def KL_div(pi_s, pi_z):
    # pi_s[a,s], pi_z[a,z]
    
    kldiv = 0
    
    n_states = pi_s.shape[1]
    n_latent = pi_z.shape[1]
    n_actions = pi_s.shape[0]
    
    #print('n_s: ', n_states, 'n_z: ', n_latent, 'n_actions: ', n_actions)
    
    for s in range(n_states):
        for z in range(n_latent):
            for a in range(n_actions):
                kldiv += pi_z[a, z] * np.log(pi_z[a,z]/pi_s[a,s])
    return kldiv

def get_d_sz(pi_s,pi_z):
    # pi_s[a,s], pi_z[a,z]
    
    n_states = pi_s.shape[1]
    n_latent = pi_z.shape[1]
    n_actions = pi_s.shape[0]
    
    d_sz = np.zeros((n_states, n_latent))
    for s in range(n_states):
        for z in range(n_latent):
            for a in range(n_actions):
                d_sz[s,z] += pi_z[a, z] * np.log(pi_z[a,z]/pi_s[a,s])
    return d_sz

def get_d_sz2(pi_s,pi_z):
    # pi_s[a,s], pi_z[a,z]
    
    n_states = pi_s.shape[1]
    n_latent = pi_z.shape[1]
    n_actions = pi_s.shape[0]
    
    d_sz = np.zeros((n_states, n_latent))
    for s in range(n_states):
        for z in range(n_latent):
            for a in range(n_actions):
                d_sz[s,z] += pi_s[a, s] * np.log(pi_s[a,s]/pi_z[a,z])
    return d_sz

def normalize_dist(dist):
    return dist/np.sum(dist)


def run_IB(n_iters, pi_z_init, n_steps, pi_s):
    
    # global variables
    n_states = 4 # definition of s
    n_latent = 2 # definition of z
    n_actions = 2 # definition of a
    beta = 2

    # iterative functions
    p_s = 1/n_states - 0.001 # uniform probability for each state - added help of normalization
    pi_z_h = []
    
    for n in range(n_iters):
        
        p_z = np.array([0.5,0.5])
        p_z = normalize_dist(p_z)
        pi_z = pi_z_init
        d_sz = np.zeros((n_states, n_latent))
        Z_sb = np.zeros((n_states))
        rho_zs = np.zeros((n_latent, n_states))

        for t in range(n_steps):

            d_sz = get_d_sz(pi_s, pi_z)

            # update Z_sb
            Z_sb = np.zeros((n_states))
            for s in range(n_states):
                for z in range(n_latent):
                    Z_sb[s] += p_z[z] * np.exp(-beta * d_sz[s,z])

            # update rho_zs
            for s in range(n_states):
                for z in range(n_latent):
                    rho_zs[z,s] = (p_z[z]/Z_sb[s]) * np.exp(-beta * d_sz[s,z])

            # update p_z
            p_z = np.zeros(n_latent)
            for z in range(n_latent):
                for s in range(n_states):
                    p_z[z] += rho_zs[z,s] * p_s

            # update pi_z
            pi_z = np.zeros((n_actions, n_latent))
            for a in range(n_actions):
                for z in range(n_latent):
                    for s in range(n_states):
                        pi_z[a,z] +=  p_s * pi_s[a,s] * rho_zs[z,s] / p_z[z]
        
            pi_z_h.append(pi_z)

        pi_z_h = np.asarray(pi_z_h)
        
        return np.asarray(pi_z_h), rho_zs

#%%

# global variables
n_states = 4 # definition of s
n_latent = 2 # definition of z
n_actions = 2 # definition of a

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

fig = plt.figure(figsize = [7,7])
gs = GridSpec(4, 2)
ax = []
for i,e in enumerate(pi_z_inits_ds):
    ax.append([])
    ax[i] = fig.add_subplot(gs[i])
    ax[i].imshow(e, 'Blues')
    ax[i].tick_params(labelbottom=False, labelleft=False, bottom = False, left = False)
plt.show()

#%%

# initialize pi_s

# normalizing policies over actions 
pi_s = 0.001 * np.ones((n_actions, n_states))
pi_s[0,0:int(n_states/2)] = 1
pi_s[1,int(n_states/2):n_states] = 1
pi_s = pi_s / pi_s.sum(axis=0)

plt.imshow(pi_s, 'Blues')

# run the IB over multiple initial conditons


pi_z_sol = []
rho_zs_cond = np.zeros((len(pi_z_inits_ds), n_latent, n_states))

for i,e in enumerate(pi_z_inits_ds):
    
    pi_z_h, rho_zs = run_IB(20, e, 20, pi_s)
    pi_z_sol.append(pi_z_h[-1])
    rho_zs_cond[i] = rho_zs
    
    fig = plt.figure(figsize = [20,3])
    gs = GridSpec(1,5)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[4])
    
    ax1.imshow(e, 'Blues')
    ax1.tick_params(labelbottom=False, labelleft=False, bottom = False, left = False)
    
    ax2.plot(pi_z_h[:,0,0], 'b', label = 'z = 0, a = 0')
    ax2.plot(pi_z_h[:,0,1], 'b--', label = 'z= 0, a = 1')
    ax2.plot(pi_z_h[:,1,0], 'r', label = 'z = 1, a = 0')
    ax2.plot(pi_z_h[:,1,1], 'r--',label = 'z = 1, a = 1')
    ax2.set_xlabel('step')
    ax2.set_title('$\pi_z$')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.legend()
    
    ax3.set_title('$\pi_z$')
    ax3.imshow(pi_z_h[-1], 'Blues')
    ax3.tick_params(labelbottom=False, labelleft=False, bottom = False, left = False)
    
    ax4.set_title('$p_{zs}$')
    ax4.imshow(rho_zs, 'Blues')
    ax4.tick_params(labelbottom=False, labelleft=False, bottom = False, left = False)
    
    pi_s_calc = np.zeros((n_actions, n_states))
    for a in range(n_actions):
        for s in range(n_states):
            for z in range(n_latent):
                pi_s_calc[a,s] += rho_zs_cond[1,z,s].T * pi_z_h[-1,a,z]
    
    ax5.set_title('$\pi_s$')
    ax5.imshow(pi_s_calc, 'Blues')
    ax5.tick_params(labelbottom=False, labelleft=False, bottom = False, left = False)
    
    #print(check_pi_solution(pi_z_t))
    plt.show()

pi_z_sol = np.asarray(pi_z_sol)
# %%
