import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.gridspec import GridSpec

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
        k += p_s[s] * kl(pi_behave[s,:], pi_s[s,:])
    return k

## information bottleneck functions - two versions

def run_information_bottleneck(rho, pi_z_init, n_steps, beta):

    # setup: four environment states, two latent states, two actions
    
    n_states = 4
    n_latent = 2
    n_actions = 2
    
    
    pi_z = pi_z_init
    
    pi_behave = np.ones((n_actions, n_states))/n_actions # behavioral policy induced throgh latent representation
    pi_s = np.array([[0.99, 0.99, 0.01, 0.01], [0.01, 0.01, 0.99, 0.99]]) # target policy
    p_z = np.ones((1, n_latent))/n_latent
    p_s = np.ones((1, n_states))/n_states
    p_sz = np.ones((n_states, n_latent))/n_latent

    error_behave = [error(p_s, pi_behave, pi_s)]
    pi_z_h = []

    for t in range(10):
        for s in range(n_states):
            for z in range(n_latent):
                rho[z,s] = p_z[0,z] * np.exp(-beta * kl(pi_s[:,s], pi_z[:,z])) 
            rho[:,s] /= (rho[:,s]).sum()
            
        pi_z_h.append(pi_z)

        # compute marginal latents
        
        p_z = (p_s * rho).sum(1).reshape(1, n_latent)
        
        # compute environment state
        p_sz = (rho * p_s)
        p_sz /= (p_sz.sum(1)).reshape(n_latent, 1)
        #print('p_sz', p_sz.shape)
        
        # compute latent policy
        pi_z = np.dot(p_sz, pi_s.T)

        # compute behavioral policy
        pi_behave = np.dot(rho.T, pi_z).T

        # record KL between target policy and behavioral policy
        error_behave.append(error(p_s, pi_behave, pi_s))

    return {'rho':rho, 'pi_z':pi_z, 'p_sz':p_sz, 
            'pi_behave':pi_behave, 'error_behave':error_behave[0:-1],
            'pi_s':pi_s, 'pi_z_h' : np.asarray(pi_z_h)}


## plotting 

def plot_sim_metrics(sim_data, fig_size):
    
    rho = sim_data['rho']
    pi_z = sim_data['pi_z']
    n_steps = sim_data['pi_z_h'].shape[0]
    
    fig, axes = plt.subplots(1,5, figsize = fig_size)
    vmin = 0; vmax = 1.; cmap = 'Greys'

    axes[0].imshow(sim_data['pi_z_h'][0], vmin=vmin, vmax=vmax, cmap=cmap)  
    axes[0].set(ylabel = 'action', xlabel = 'latent state (z)', xticks = [0,1], yticks = [0,1], title = 'initial $\pi_z$')
    axes[1].plot(range(n_steps), sim_data['error_behave'], 'k')
    axes[1].set(ylabel = 'KL', xlabel = 'iterations', yticks = [0,1], aspect=10)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[2].imshow(sim_data['pi_z'], vmin=vmin, vmax=vmax, cmap=cmap)
    axes[2].set(ylabel = 'action', xlabel = 'latent state (z)', xticks = [0,1], yticks = [0,1], title = 'learned $\pi_z$')
    axes[3].imshow(sim_data['rho'].T, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[3].set(xlabel = 'action', ylabel = 'observed state (s)', xticks = [0,1], yticks = [0,1,2,3], title = r'$\rho(z|s)$', aspect = 0.5)
    axes[4].imshow(np.dot(rho.T,pi_z), vmin=vmin, vmax=vmax, cmap=cmap)
    axes[4].set(xlabel = 'action', ylabel = 'observed state (s)', xticks = [0,1], yticks = [0,1,2,3], title = 'reconstructed $\pi_s$', aspect = 0.5)

    plt.show()
    
    
    ## initial conditions
    
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
    pi_z /= pi_z.sum(axis=0)
    pi_z_inits_ds.append(pi_z)
    rho_inits.append(rho_opt)

    pi_z = np.zeros((n_actions, n_latent))
    pi_z[0,0] = 0.02
    pi_z[0,1] = 0.01
    pi_z[1,0] = 0.01
    pi_z[1,1] = 0.01
    pi_z /= pi_z.sum(axis=0)
    pi_z_inits_ds.append(pi_z)
    rho_inits.append(rho_opt)

    pi_z = np.zeros((n_actions, n_latent))
    pi_z[0,0] = 0.01
    pi_z[0,1] = 0.02
    pi_z[1,0] = 0.01
    pi_z[1,1] = 0.01
    pi_z /= pi_z.sum(axis=0)
    pi_z_inits_ds.append(pi_z)
    rho_inits.append(rho_conf)

    pi_z = np.zeros((n_actions, n_latent))
    pi_z[0,0] = 0.01
    pi_z[0,1] = 0.01
    pi_z[1,0] = 0.02
    pi_z[1,1] = 0.01
    pi_z /= pi_z.sum(axis=0)
    pi_z_inits_ds.append(pi_z)
    rho_inits.append(rho_conf)

    pi_z = np.zeros((n_actions, n_latent))
    pi_z[0,0] = 0.01
    pi_z[0,1] = 0.01
    pi_z[1,0] = 0.01
    pi_z[1,1] = 0.02
    pi_z /= pi_z.sum(axis=0)
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
    pi_z /= pi_z.sum(axis=0)
    pi_z_inits_ds.append(pi_z)
    rho_inits.append(rho_opt)

    pi_z = np.zeros((n_actions, n_latent))
    pi_z[0,0] = 0.02
    pi_z[0,1] = 0.01
    pi_z[1,0] = 0.01
    pi_z[1,1] = 0.02
    pi_z /= pi_z.sum(axis=0)
    pi_z_inits_ds.append(pi_z)
    rho_inits.append(rho_opt)

    pi_z = np.zeros((n_actions, n_latent))
    pi_z[0,0] = 0.01
    pi_z[0,1] = 0.02
    pi_z[1,0] = 0.02
    pi_z[1,1] = 0.01
    pi_z /= pi_z.sum(axis=0)
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
    