import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class SARSA_agent():
    
    def __init__(self, state_size, action_size, alpha, gamma, beta):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        
        # state-action value function
        self.Q = np.random.uniform(0,0.001, [state_size, action_size]) # action preference

    def sample_action(self, current_state):
        
        # softmax
        x = self.Q[current_state,:] - self.Q[current_state,:].max(axis=None, keepdims=True)
        y = np.exp(x * self.beta)
        action_prob = y / y.sum(axis=None, keepdims=True)
        
        return np.argmax(np.random.multinomial(1, action_prob, size=1))
    
    def get_V(self, grid_size):
        
        v = np.zeros([self.state_size])
        
        for i in range(self.state_size):
            v[i] =  np.mean(self.Q[i,:])
        
        return np.reshape(v, [grid_size, grid_size])

    def update_td(self, experience):
        
        s = experience[0] # current state
        a = experience[1] # current action
        s1 = experience[2] # next state
        a1 = experience[3] # next action
        r = experience[4] # reward
        
        # update td-error
        delta = r + self.gamma * self.Q[s1,a1] - self.Q[s,a]
        
        # update Q
        self.Q[s,a] += self.alpha * delta
        
        return delta


# SUCCESSOR AGENT

class SuccessorAgent():
    
    def __init__(self, state_size, action_size, learning_rate, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.M = np.stack([np.identity(state_size) for i in range(action_size)])
        self.w = np.zeros([state_size])
        self.learning_rate = learning_rate
        self.gamma = gamma
        
    def onehot(self, value, max_value):
        vec = np.zeros(max_value)
        vec[value] = 1
        return vec
        
    def Q_estimates(self, state, goal=None):
        # Generate Q values for all actions.
        if goal == None:
            goal = self.w
        else:
            goal = self.onehot(goal, self.state_size)
        return np.matmul(self.M[:,state,:],goal)
    
    def sample_action(self, state, goal=None, epsilon=0.0):
        
        # Samples action using epsilon-greedy approach
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(self.action_size)
        else:
            Qs = self.Q_estimates(state, goal)
            action = np.argmax(Qs)
            
        return action
    
    def update_w(self, current_exp):
        
        # A simple update rule
        s_1 = current_exp[2]
        r = current_exp[3]
        error = r - self.w[s_1]
        self.w[s_1] += self.learning_rate * error
        
        return error
    
    # this part is a bit weirdly programmed
    def update_sr(self, current_exp, next_exp):
        # SARSA TD learning rule
        s = current_exp[0]
        s_a = current_exp[1]
        s_1 = current_exp[2]
        s_a_1 = next_exp[1]
        r = current_exp[3]
        d = current_exp[4]
        I = self.onehot(s, self.state_size)
        
        if d: #done     
            td_error = (I + self.gamma * self.onehot(s_1, self.state_size) - self.M[s_a, s, :])
        else: # not done
            td_error = (I + self.gamma * self.M[s_a_1, s_1, :] - self.M[s_a, s, :])
            
        self.M[s_a, s, :] += self.learning_rate * td_error
        
        return td_error
    
# SINGLE ACTOR-CRITIC

class SingleActorCritic():
    
    def __init__(self, state_size, action_size, gamma, beta):
        
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = np.ones(state_size) * 0.1
        self.sv = np.zeros(state_size)
        self.gamma = gamma
        self.beta = beta
        self.alpha_c = 0.01
        
        self.V = np.zeros([state_size]) # state value function
        self.A = np.random.uniform(0,0.01, [state_size, action_size]) # advantage 

    def sample_action(self, At, beta):
        # softmax
        x = At - At.max(axis = None, keepdims = True)
        y = np.exp(x * self.beta)
        action_prob = y / y.sum(axis = None, keepdims = True)
        
        return np.argmax(np.random.multinomial(1, action_prob, size=1))
    
    # make this plot a bit prettier
    def plot_transfer(self):
        x = np.linspace(-10,10,1000)
        plt.plot(x,self.nF(x,1))
        plt.plot(x,self.nF(x,-1))
        plt.show()
    
    def update_td(self, current_exp):
        
        s = current_exp[0] # current state
        a = current_exp[1] # current action
        s1 = current_exp[2] # next state
        r = current_exp[3] # reward
        t_flag = current_exp[4] 
        
        # update td-error
        
        if t_flag == False:
            delta = r + self.gamma * self.V[s1] - self.V[s]
        else:
            delta = r - self.V[s]
        
        # update critic
        self.V[s] += self.alpha_c * delta
        
        # update actors
        self.A[s,a] += self.alpha_c * delta # direct pathway actor
        
        return delta

# MULTI-ACTOR CRITIC

class MultiActorCritic():
    
    def __init__(self, state_size, action_size, gamma, beta, wA):
        
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = np.ones(state_size) * 0.1
        self.alpha_c = 0.05
        self.sv = np.zeros(state_size)
        self.gamma = gamma
        self.beta = beta
        
        self.n_p = 2 # number of reward pathways // direct indirect only
        # nlinear trasfer function parameters
        self.pa = 0
        self.pb = 1
        self.pc = 3.7
        
        self.V = np.zeros([state_size]) # state value function
        self.A = np.random.uniform(0,0.01, [self.n_p, state_size, action_size]) # action preference
        self.tA = np.zeros([state_size, action_size]) # total action preference
        self.wA = wA
        
        self.tter = []

    def sample_action(self, tA, beta):
        # softmax
        x = tA - tA.max(axis=None, keepdims=True)
        y = np.exp(x * self.beta)
        action_prob = y / y.sum(axis=None, keepdims=True)
        
        return np.argmax(np.random.multinomial(1, action_prob, size=1))
    
    # nonlinear transfer function for the actors
    def nF(self, x, p):
        
        return self.pa + self.pb / (1 + self.pc * np.exp(1 - p * x))
    
    # make this plot a bit prettier
    def plot_transfer(self):
        x = np.linspace(-10,10,1000)
        plt.plot(x,self.nF(x,1))
        plt.plot(x,self.nF(x,-1))
        plt.show()
    
    def update_td(self, current_exp):
        
        s = current_exp[0] # current state
        a = current_exp[1] # current action
        s1 = current_exp[2] # next state
        r = current_exp[3] # reward
        tf = current_exp[4]

        if t_flag == False:
            delta = r + self.gamma * self.V[s1] - self.V[s]

        if t_flag == True:
            delta = r - self.V[s]
        
        # update critic
        self.V[s] += self.alpha_c * delta
        
        # update actors
        self.A[0,s,a] += self.alpha_c * self.nF(delta, 1) # direct pathway actor
        self.A[1,s,a] += self.alpha_c * self.nF(delta, -1) # indirect pathway actor
        
        self.tA = wA[0] * self.A[0,:,:] + wA[1] * self.A[1,:,:]
        
        return delta
    

        
