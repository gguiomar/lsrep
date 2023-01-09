#%%
# General imports.
import torch
import torch.nn
import matplotlib.pyplot as plt

# Planar flow imports. 
from planar_flow import PlanarFlow
from target_distribution import TargetDistribution
from loss import VariationalLoss

# visualization tools
from utils.gif import make_gif_from_train_plots
from utils.plot import plot_training, plot_density, plot_available_distributions, plot_comparison

#%%

# Plot available distributions.
plot_available_distributions()

#%%

# Parameters.
target_distr = "U_1"  # The distribution we want to learn. Choices: ["U_1", "U_2", "U_3", "U_4", "ring"].
flow_length = 32  # Number of transformations in the flow. 
num_batches = 5000  # How many training batches to train for. These are generated on the fly. 
batch_size = 128  # This is... wait for it... the size of each batch. 
lr = 6e-4  # The learning rate for the optimiser.
axlim = 7  # Setting for plotting. Recommended: 5 for 'U_1' to 'U_4', 7 for 'ring'.

#%%

# Initialise model, loss, and optimiser. 
model = PlanarFlow(dim=2, K=flow_length)
density = TargetDistribution(target_distr)
bound = VariationalLoss(density)
optimiser = torch.optim.Adam(model.parameters(), lr=lr)

#%%

# Train model. 
loss_h = []
for batch_num in range(1, num_batches + 1):
    # Get batch from N(0,I).
    batch = torch.zeros(size=(batch_size, 2)).normal_(mean=0, std=1)
    # Pass batch through flow.
    zk, log_jacobians = model(batch)
    # Compute loss under target distribution.
    loss = bound(batch, zk, log_jacobians)
    loss_h.append(loss.detach())
    
    # Train. 
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if batch_num % 100 == 0:
        print(f"(batch_num {batch_num:05d}/{num_batches}) loss: {loss}")

    if batch_num == 1 or batch_num % 100 == 0:
        # Save plots during training. Plots are saved to the 'train_plots' folder.
        plot_training(model, flow_length, batch_num, lr, axlim)

plt.plot(loss_h)

#%%
     
# Plot true and estimated denisty side by side. 
plot_comparison(model, target_distr, flow_length)

#%%
# Generate and display an animation of the training progress.
make_gif_from_train_plots('notebook.gif')

# %%
