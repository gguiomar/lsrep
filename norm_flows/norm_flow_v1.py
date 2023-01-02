import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Lambda, FlowNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Generate a bimodal dataset
np.random.seed(1)
data = np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)])

# Create a normalizing flows model
inputs = Input(shape=(1,))
z = Dense(20, activation='relu')(inputs)
z = Dense(20, activation='relu')(z)
z = Dense(20, activation='relu')(z)
z = Dense(2)(z)
z = FlowNormalization()(z)
model = Model(inputs=inputs, outputs=z)

# Define the loss function and the optimizer
def neg_log_likelihood(y_true, y_pred):
  return -tf.reduce_mean(y_pred.log_prob(y_true))

optimizer = Adam(learning_rate=0.01)

# Compile the model
model.compile(optimizer=optimizer, loss=neg_log_likelihood)

# Fit the model to the data
model.fit(data.reshape(-1, 1), epochs=100, verbose=False)

# Plot the data and the fitted normalizing flows model
plt.hist(data, bins=20, density=True)
x = np.linspace(-5, 10, 1000)
logprob = model(x.reshape(-1, 1)).log_prob(x.reshape(-1, 1))
pdf = np.exp(logprob)
plt.plot(x, pdf, '-k')
plt.show()