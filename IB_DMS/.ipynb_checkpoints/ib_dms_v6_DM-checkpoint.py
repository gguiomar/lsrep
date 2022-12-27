#!/usr/bin/python

# My take-away intuitions:
# 1. Need noise in latent policy initialization otherwise stuck on nullcline (i.e. algorithm can't "decide" which way to go)
# 2. Increase beta for lower error and faster convergence

import numpy as np
import matplotlib.pyplot as plt

# setup: four environment states, two latent states, two actions
n_state = 4
n_latent = 2
n_action = 2

pi_behave = np.ones((n_state, n_action))/n_action # behavioral policy induced through latent representation
pi_s = np.array([[0.99, 0.01], [0.99, 0.01], [0.01, 0.99], [0.01, 0.99]]) # target policy
rho = np.ones((n_state, n_latent))/n_latent
pi_z = np.ones((n_latent, n_action))/n_action
p_z = np.ones((1, n_latent))/n_latent
p_s = np.ones((n_state, 1))/n_state
p_sz = np.ones((n_latent, n_state))/n_latent

def kl(p1,p2):
    return (p1*np.log(p1/p2)).sum()

def error(pi_behave, pi_s):
    k = 0.
    for s in range(n_state):
        k += p_s[s]*kl(pi_behave[s,:], pi_s[s,:])
    return k

n_steps = 10
beta = 100.

# case 0 [DOES NOT WORK]:
# flat priors, doesn't work, stuck on nullcline

# case 1 [WORKS]: random latent policy initialization
# pi_z += np.random.normal(1,0.1, size=(n_latent, n_action))
# pi_z /= pi_z.sum(axis=1)

# case 2 [WORKS]: deliberate confusion, behavioral policy reversed from target policy
# rho maps states 0,1 to latent 1, 2,3 -> 0
# latent policy maps latent state 0 to action 0, 1->1
# implies states 0,1 mapped to action 1, and 2,3->0 (opposite to target)
rho = np.array([[.1, .9], [.1, .9], [.9, .1], [.9, .1]])
pi_z = np.array([[0.99, 0.01], [0.01, 0.99]])

error_behave = [error(pi_behave, pi_s)]
for t in range(n_steps):
    for s in range(n_state):
        for z in range(n_latent):
            rho[s,z] = p_z[0, z]*np.exp(-beta*kl(pi_s[s,:], pi_z[z,:]))
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
    error_behave.append(error(pi_behave, pi_s))

plt.plot(range(n_steps+1), error_behave)
plt.xlabel('Number of iterations')
plt.ylabel('KL[behavior, target policy]')

fig, axes = plt.subplots(1,2)
vmin = 0; vmax = 1.; cmap = 'Greys'
axes[0].imshow(pi_behave, vmin=vmin, vmax=vmax, cmap=cmap)
axes[0].set_title('Behavior')
axes[1].imshow(pi_s, vmin=vmin, vmax=vmax, cmap=cmap)
axes[1].set_title('Target policy')

fig, axes = plt.subplots(1,2)
axes[0].imshow(rho, vmin=vmin, vmax=vmax, cmap=cmap)
axes[0].set_title('rho(z|s)')
axes[1].imshow(p_sz, vmin=vmin, vmax=vmax, cmap=cmap)
axes[1].set_title('p(s|z)')

fig, axes = plt.subplots(1,2)
axes[0].imshow(p_s, vmin=vmin, vmax=vmax, cmap=cmap)
axes[0].set_title('p(s)')
axes[1].imshow(p_z, vmin=vmin, vmax=vmax, cmap=cmap)
axes[1].set_title('p(z)')

plt.show()