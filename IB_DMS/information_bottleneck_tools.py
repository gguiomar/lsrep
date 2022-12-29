import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "20"


# auxiliary functions DM

def kl(p1,p2):
    return (p1*np.log(p1/p2)).sum()

def error(p_s, pi_behave, pi_s):
    k = 0.
    n_state = p_s.shape[0]
    for s in range(n_state):
        k += p_s[s]*kl(pi_behave[s,:], pi_s[s,:])
    return k
    

## auxiliary function GG

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

## information bottleneck functions - two versions

def run_information_bottleneck_DM(rho, pi_z_init, n_steps, beta):

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
    pi_z_h = [pi_z_init]
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
        pi_z_h.append(pi_z)

        # compute behavioral policy
        pi_behave = np.dot(rho, pi_z)

        # record KL between target policy and behavioral policy
        error_behave.append(error(p_s, pi_behave, pi_s))

    return {'rho':rho, 'pi_z':pi_z, 'p_sz':p_sz, 
            'pi_behave':pi_behave, 'error_behave':error_behave,
            'pi_s':pi_s, 'pi_z_h' : np.asarray(pi_z_h)}

def run_information_bottleneck_GG(rho, pi_z_init, n_steps, beta):
    
    # global variables
    n_states = 4 # definition of s
    n_latent = 2 # definition of z
    n_actions = 2 # definition of a
    beta = 100

    # iterative functions
    p_s = 1/n_states - 0.001 # uniform probability for each state - added help of normalization
    pi_z_h = []
        
    p_z = np.array([0.5,0.5])
    p_z = normalize_dist(p_z)
    pi_z = pi_z_init
    d_sz = np.zeros((n_states, n_latent))
    Z_sb = np.zeros((n_states))
    pi_s = np.array([[0.99, 0.01], [0.99, 0.01], [0.01, 0.99], [0.01, 0.99]]).T # target policy
    #rho = np.zeros((n_latent, n_states))

    for t in range(n_steps):

        d_sz = get_d_sz(pi_s, pi_z)

        # update Z_sb
        Z_sb = np.zeros((n_states))
        for s in range(n_states):
            for z in range(n_latent):
                Z_sb[s] += p_z[z] * np.exp(-beta * d_sz[s,z])

        # update rho
        for s in range(n_states):
            for z in range(n_latent):
                rho[z,s] = (p_z[z]/Z_sb[s]) * np.exp(-beta * d_sz[s,z])

        # update p_z
        p_z = np.zeros(n_latent)
        for z in range(n_latent):
            for s in range(n_states):
                p_z[z] += rho[z,s] * p_s

        # update pi_z
        pi_z = np.zeros((n_actions, n_latent))
        for a in range(n_actions):
            for z in range(n_latent):
                for s in range(n_states):
                    pi_z[a,z] +=  p_s * pi_s[a,s] * rho[z,s] / p_z[z]
    
        pi_z_h.append(pi_z)

    pi_z_h = np.asarray(pi_z_h)
    
    return {'pi_z': pi_z_h[-1], 'pi_z_h': np.asarray(pi_z_h), 'rho':rho}


## plotting 

def plot_rho_pi_comparison(sim_data_opt, sim_data_conf):
    
    rho_o = sim_data_opt['rho']
    rho_c = sim_data_conf['rho']
    pi_z_o = sim_data_opt['pi_z']
    pi_z_c = sim_data_conf['pi_z']
    n_steps = sim_data_opt['pi_z_h'].shape[0]

    fig, axes = plt.subplots(2,4, figsize = (20,17))
    vmin = 0; vmax = 1.; cmap = 'Greys'

    axes[0,0].plot(range(n_steps), sim_data_opt['error_behave'], 'k')
    axes[0,0].set(ylabel = 'KL', xlabel = 'iterations', title = 'Optimal IC', yticks = [0,1], aspect=6)
    axes[0,0].spines['right'].set_visible(False)
    axes[0,0].spines['top'].set_visible(False)
    axes[0,1].imshow(sim_data_opt['pi_z'], vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0,1].set(ylabel = 'action', xlabel = 'latent state (z)', xticks = [0,1], yticks = [0,1], title = 'learned $\pi_z$')
    axes[0,2].imshow(sim_data_opt['rho'], vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0,2].set(xlabel = 'action', ylabel = 'observed state (s)', xticks = [0,1], yticks = [0,1,2,3], title = r'$\rho(z|s)$', aspect = 0.5)
    axes[0,3].imshow(np.dot(rho_o,pi_z_o), vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0,3].set(xlabel = 'action', ylabel = 'observed state (s)', xticks = [0,1], yticks = [0,1,2,3], title = 'reconstructed $\pi_s$', aspect = 0.5)

    axes[1,0].plot(range(n_steps), sim_data_conf['error_behave'], 'k')
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
    

def plot_sim_metrics(sim_data, fig_size):
    
    rho = sim_data['rho']
    pi_z = sim_data['pi_z']
    n_steps = sim_data['pi_z_h'].shape[0]
    
    fig, axes = plt.subplots(1,5, figsize = fig_size)
    vmin = 0; vmax = 1.; cmap = 'Greys'

    
    axes[0].imshow(sim_data['pi_z_h'][0], vmin=vmin, vmax=vmax, cmap=cmap)  
    axes[0].set(ylabel = 'action', xlabel = 'latent state (z)', xticks = [0,1], yticks = [0,1], title = 'initial $\pi_z$')
    
    axes[1].plot(range(n_steps), sim_data['error_behave'], 'k')
    axes[1].set(ylabel = 'KL', xlabel = 'iterations', title = 'Optimal IC', yticks = [0,1], aspect=6)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    
    axes[2].imshow(sim_data['pi_z'], vmin=vmin, vmax=vmax, cmap=cmap)
    axes[2].set(ylabel = 'action', xlabel = 'latent state (z)', xticks = [0,1], yticks = [0,1], title = 'learned $\pi_z$')
    
    axes[3].imshow(sim_data['rho'], vmin=vmin, vmax=vmax, cmap=cmap)
    axes[3].set(xlabel = 'action', ylabel = 'observed state (s)', xticks = [0,1], yticks = [0,1,2,3], title = r'$\rho(z|s)$', aspect = 0.5)

    axes[4].imshow(np.dot(rho,pi_z), vmin=vmin, vmax=vmax, cmap=cmap)
    axes[4].set(xlabel = 'action', ylabel = 'observed state (s)', xticks = [0,1], yticks = [0,1,2,3], title = 'reconstructed $\pi_s$', aspect = 0.5)