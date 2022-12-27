#!/usr/bin/python

# My take-away intuitions:
# 1. Need noise in latent policy initialization otherwise stuck on nullcline (i.e. algorithm can't "decide" which way to go)
# 2. Increase beta for lower error and faster convergence

#%% inits

import numpy as np
import matplotlib.pyplot as plt

def kl(p1,p2):
    return (p1*np.log(p1/p2)).sum()

def error(p_s, pi_behave, pi_s):
    k = 0.
    for s in range(n_state):
        k += p_s[s]*kl(pi_behave[s,:], pi_s[s,:])
    return k

def plot_rho_pi(rho, pi_z, title):
    fig, axes = plt.subplots(1,2)
    if title:
        fig.suptitle(title)
    vmin = 0; vmax = 1.; cmap = 'Greys'
    axes[0].imshow(rho, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0].set_title('rho')
    axes[1].imshow(pi_z, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[1].set_title('pi(z)')
    plt.show()

n_state = 4
n_latent = 2
n_action = 2

n_steps = 10
beta = 100

# rho = np.ones((n_state, n_latent))/n_latent
# pi_z = np.ones((n_latent, n_action))/n_action

# case 0 [DOES NOT WORK]:
# flat priors, doesn't work, stuck on nullcline

# case 1 [WORKS]: random latent policy initialization
# pi_z += np.random.normal(1,0.1, size=(n_latent, n_action))
# pi_z /= pi_z.sum(axis=1)

# case 2 [WORKS]: deliberate confusion, behavioral policy reversed from target policy
# rho maps states 0,1 to latent 1, 2,3 -> 0
# latent policy maps latent state 0 to action 0, 1->1
# implies states 0,1 mapped to action 1, and 2,3->0 (opposite to target)

def run_information_bottleneck(rho, pi_z_init):

    # setup: four environment states, two latent states, two actions
    
    n_state = 4
    n_latent = 2
    n_action = 2
    
    pi_z = pi_z_init
    pi_behave = np.ones((n_state, n_action))/n_action # behavioral policy induced through latent representation
    pi_s = np.array([[0.99, 0.01], [0.99, 0.01], [0.01, 0.99], [0.01, 0.99]]) # target policy
    p_z = np.ones((1, n_latent))/n_latent
    p_s = np.ones((n_state, 1))/n_state
    p_sz = np.ones((n_latent, n_state))/n_latent

    error_behave = []
    error_behave = [error(p_s, pi_behave, pi_s)]
    for t in range(n_steps):
        for s in range(n_state):
            for z in range(n_latent):
                rho[s,z] = p_z[0, z]*np.exp(-beta*kl(pi_s[s,:], pi_z[z,:])) #check this implementation
            rho[s,:] /= (rho[s,:]).sum()

        # compute marginal latents
        p_z = (p_s*rho).sum(0).reshape(1, n_latent)

        # compute environment state
        p_sz = (p_s*rho).T
        p_sz /= (p_sz.sum(1)).reshape(n_latent,1)

        # compute latent policy
        pi_z = np.dot(p_sz, pi_s)

        # compute behavioral policy
        pi_behave = np.dot(rho, pi_z)

        # record KL between target policy and behavioral policy
        error_behave.append(error(p_s, pi_behave, pi_s))

    return {'rho':rho, 'pi_z':pi_z, 'p_sz':p_sz, 
            'pi_behave':pi_behave, 'error_behave':error_behave,
            'pi_s':pi_s}


#%%

# optimal initial conditions ---

rho = np.array([[.9, .1], [.9, .1], [.1, .9], [.1, .9]])
pi_z_init_opt = np.array([[0.99, 0.01], [0.01, 0.99]])
plot_rho_pi(rho, pi_z_init_opt, 'OIC_init')

sim_data_opt = run_information_bottleneck(rho, pi_z_init_opt)
plot_rho_pi(sim_data_opt['rho'], sim_data_opt['pi_z'], 'OIC_conv')

plt.plot(range(n_steps+1), sim_data_opt['error_behave'])
plt.xlabel('Number of iterations')
plt.ylabel('KL[behavior, target policy]')
plt.show()

fig, axes = plt.subplots(1,2)
fig.suptitle('Policy comparison OIC')
vmin = 0; vmax = 1.; cmap = 'Greys'
axes[0].imshow(sim_data_opt['pi_behave'], vmin=vmin, vmax=vmax, cmap=cmap)
axes[0].set_title('Behavior')
axes[1].imshow(sim_data_opt['pi_s'], vmin=vmin, vmax=vmax, cmap=cmap)
axes[1].set_title('Target policy')
plt.show()


# deliberate confusion ----

rho = np.array([[.1, .9], [.1, .9], [.9, .1], [.9, .1]])
pi_z_init_conf = np.array([[0.01, 0.99],[0.99, 0.01]])
plot_rho_pi(rho, pi_z_init_conf, 'DC_init')

sim_data_conf = run_information_bottleneck(rho, pi_z_init_conf)
plot_rho_pi(sim_data_conf['rho'], sim_data_conf['pi_z'], 'DC_conv')

plt.plot(range(n_steps+1), sim_data_conf['error_behave'])
plt.xlabel('Number of iterations')
plt.ylabel('KL[behavior, target policy]')
plt.show()

fig, axes = plt.subplots(1,2)
vmin = 0; vmax = 1.; cmap = 'Greys'
fig.suptitle('Policy comparison DC')
axes[0].imshow(sim_data_conf['pi_behave'], vmin=vmin, vmax=vmax, cmap=cmap)
axes[0].set_title('Behavior')
axes[1].imshow(sim_data_conf['pi_s'], vmin=vmin, vmax=vmax, cmap=cmap)
axes[1].set_title('Target policy')
plt.show()


# %%

rho_o = sim_data_opt['rho']
rho_c = sim_data_conf['rho']
pi_z_o = sim_data_opt['pi_z']
pi_z_c = sim_data_conf['pi_z']

plt.imshow(np.dot(rho_o,pi_z_o),'Greys')
plt.show()
plt.imshow(np.dot(rho_c,pi_z_c),'Greys')
plt.show()


# %%

fig, axes = plt.subplots(2,3)
vmin = 0; vmax = 1.; cmap = 'Greys'
axes[0,0].imshow(sim_data_conf['pi_z'], vmin=vmin, vmax=vmax, cmap=cmap)
axes[0,0].set_title('DC')
axes[0,1].imshow(sim_data_conf['rho'], vmin=vmin, vmax=vmax, cmap=cmap)
axes[0,2].imshow(np.dot(rho_c,pi_z_c), vmin=vmin, vmax=vmax, cmap=cmap)

axes[1,0].imshow(sim_data_opt['pi_z'], vmin=vmin, vmax=vmax, cmap=cmap)
axes[1,0].set_title('OIC')
axes[1,1].imshow(sim_data_opt['rho'], vmin=vmin, vmax=vmax, cmap=cmap)
axes[1,2].imshow(np.dot(rho_o,pi_z_o), vmin=vmin, vmax=vmax, cmap=cmap)
plt.show()

# %%
