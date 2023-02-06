#%% 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import mx_env as mxe
import mx_agents as mxa

from matplotlib import rc
plt.rcParams['font.size'] = '13'
plt.rcParams.update({'font.family':'sans-serif'})

#%%
def run_sim(env, agent, n_episodes, n_steps, grid_size):
    
    cs = (0,0)
    ns = (0,0)
    action = 0
    
    state_visits = np.zeros((grid_size, grid_size))
    episodic_error = []
    experiences = []
    
    cs_st = 0
    td_v = 0

    for ep in range(n_episodes):
        cs = (0,0)
        for k in range(n_steps):
            
            # initial action and next state reward sampling
            action = agent.sample_action(cs_st)
            ns, reward, t_flag = env.get_next_state(cs, action)
            state_visits[cs[0],cs[1]] += 1

            # mapping graph id to state
            ns_st = env.node2state(ns)
            cs_st = env.node2state(cs)
            
            # sampling next action
            next_action = agent.sample_action(ns_st)

            # updating value functions
            if (k > 1) and t_flag == False:
                experiences.append(np.asarray([cs_st, action, ns_st, next_action, reward, t_flag, ep]))
                td_v = agent.update_td(experiences[-1])

            if t_flag == True:
                experiences.append(np.asarray([cs_st, action, ns_st, next_action, reward, t_flag, ep]))
                td_v = agent.update_td(experiences[-1])
                episodic_error.append(np.mean(np.abs(td_v)))
                break

            cs = ns
        
    return np.asarray(experiences), episodic_error, state_visits

#%%

grid_size = 5
env = mxe.GraphEnvGenerator(grid_size)
env.generate_open_gridworld()
env.pun = [(0,4), 0]
env.rwd = [(4,4), 10]
env.paint_nodes([3,3])

#%% 

n_episodes = 5000
n_steps = 70

node_list = list(env.G.nodes)
mSize = len(node_list)
n_actions = 4
gamma = 0.7
alpha = 0.01
beta = 0.7

sar = mxa.SARSA_agent(mSize, n_actions, alpha, gamma, beta)
xp, err, sv = run_sim(env, sar, n_episodes, n_steps, grid_size)

#%%

# select the first 2 indices of xp matrix
sa_pair = xp[:,0:2]

# add random gaussian noise to each entry of sa_pair
sa_pair = sa_pair + np.random.normal(0, 0.1, sa_pair.shape)

# plot the 2D scatter plot of sa_pair
plt.figure(figsize=(10,10))
plt.scatter(sa_pair[:,0], sa_pair[:,1], s=1)
plt.show()
# %%

# todo:
# 1. understand the way the distributions are encoded in the normalizing flow code
# 2. transform the state-action pair into that format and learn it with norm flows
# 3. abstract the code to do the same for other environments and agents
# 4. implement the single layer perceptron
# 5. understand how to extract the latent space vector from the norm flow model


# %%
