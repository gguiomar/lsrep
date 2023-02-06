import networkx as nx
import matplotlib.pyplot as plt

class GraphEnvGenerator():
    
    def __init__(self,  grid_size):
        self.grid_size = grid_size
        self.G = [] # main graph variable
        self.n_edges_removed = 0
        self.n_tries = 1
        
        # initialization of rewards and punishments
        # define multiple punishments for the lakeworld- how to implement?
        
        self.rwd = [(self.grid_size-1,self.grid_size-1), 10]
        self.pun = [(0,1), -5]
        self.terminal_state = (self.grid_size-1,self.grid_size-1)
    
    ######    ######    ######    ######    ######    ######    ######
    # GRAPH DRAWING
    def plot_graph_matrix(self):
        plt.imshow(nx.to_numpy_matrix(self.G))
        
    def draw_graph(self, fig_size):
        plt.figure(figsize=(fig_size[0], fig_size[1]))
        pos = {(x,y):(y,-x) for x,y in self.G.nodes()}
        nx.draw(self.G, pos=pos, 
                node_color='k', 
                with_labels=False,
                node_size=300)
        
    def paint_nodes(self, fig_size):
        plt.figure(figsize=(fig_size[0],fig_size[1]))

        nc = []
        for node in self.G.nodes:
            if node == self.rwd[0]:
                nc.append('g')
            if node == self.pun[0]:
                nc.append('r')
            if node != self.rwd[0] and node != self.pun[0]:
                nc.append('k')
                
                
        pos = {(x,y):(y,-x) for x,y in self.G.nodes()}

        nx.draw(self.G, pos = pos, 
                node_color = nc, 
                with_labels = False,
                node_size=300)
        
    
    ######    ######    ######    ######    ######    ######    ######
    # UTILITY FUNCTIONS FOR GRAPH GENERATION
    
    def remove_edges(self):
    
        # generates a graph with n_edges randomly removed

        for i in range(self.n_edges_removed):
            n_total_edges = len(list(self.G.edges))
            random_edge = list(self.G.edges)[np.random.randint(n_total_edges)]
            self.G.remove_edge(random_edge[0],random_edge[1])
    
    # environmnent generating functions 
    def init_state(self):
        return list(self.G.nodes)[np.random.randint(len(self.G.nodes))]
                
                
    ######    ######    ######    ######    ######    ######    ######
    # GENERATE DIFFERENT TYPES OF ENVIRONMENTS
    # Open grid world
    # Open grid world with lake (punishment space)
    # 4 ROOM (with bottlenecks)
    # 2 ROOM (with bottlenecks)
    # Maze
    
    def generate_open_gridworld(self):

        c = 0
        g_list = []
        self.n_tries = 1
        self.n_edges_removed = 0
        
        for n in range(self.n_tries):
            g = nx.grid_2d_graph(self.grid_size, self.grid_size)

        self.G = g
    
    
    def generate_maze(self):
        
        c = 0
        g_list = []
        for n in range(self.n_tries):
            g = nx.grid_2d_graph(self.grid_size, self.grid_size)
            g = self.remove_edges()
            #print(nx.is_connected(g))
            if nx.is_connected(g):
                g_list.append(g)
        
        self.G = g
    
    def generate_4_room_environment(self):
        
        self.n_tries = 1 

        f = nx.grid_2d_graph(self.grid_size, self.grid_size)
        g1 = self.generate_connected_graphs(self.grid_size, 1, self.n_tries, 0)[0]


        for i in range(grid_size):
            g1.remove_edge((4,i),(5,i))
            g1.remove_edge((i,4),(i,5))

        g1.add_edge((2,2),(5,2))
        g1.add_edge((2,4),(2,5))
        g1.add_edge((7,4),(7,5))
        g1.add_edge((4,7),(5,7))

        self.G = g1
                

    ######    ######    ######    ######    ######    ######    ######
    # STATE TRANSITION - REWARD PUNISHMENT FUNCTIONS
    
    
    def get_next_state(self, current_state, action):
        
        # given an initial state, outputs the next state after action is taken
        # need to insert rewards/punishments (probably should be a data structure)
        
        rwd_state = self.rwd[0]
        rwd_mag = self.rwd[1]       
        pun_state = self.pun[0]
        pun_mag = self.pun[1]
        next_state = 0
        t_flag = False
        
        if current_state == rwd_state:
            return current_state, rwd_mag, True
        
        else:
            reward = 0

            # get all edges of current_state node
            neigh_edges = self.G.edges(current_state)

            valid_state = False
            valid_transition = False

            if action == 0: #UP
                next_state = (current_state[0]-1, current_state[1])
            if action == 1: #DOWN
                next_state = (current_state[0]+1, current_state[1])    
            if action == 2: #LEFT
                next_state = (current_state[0], current_state[1]-1)    
            if action == 3: #RIGHT
                next_state = (current_state[0], current_state[1]+1)

            # check if next state is valid

            for node in self.G.nodes:
                if next_state == node:
                    valid_state = True

            for edge in neigh_edges:
                if next_state == edge[1]:
                    valid_transition = True

            if (valid_state & valid_transition):
                if next_state == pun_state:
                    reward = pun_mag
                    t_flag = False
                else:
                    t_flag = False

                return next_state, reward, t_flag
            else:
                reward = 0
                return current_state, reward, t_flag
    
    
    ######    ######    ######    ######    ######    ######    ######
    # NODE AND STATE TRANSFORMS
    
    def node2state(self, node):
        node_list = list(self.G.nodes)
        for i,e in enumerate(node_list):
            if e == node:
                return i