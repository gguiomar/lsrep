#%%

import numpy as np
import sklearn
from sklearn import mixture
from matplotlib import pyplot as plt
import scipy.stats as sp
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.datasets import make_blobs
from matplotlib.ticker import FixedLocator, LinearLocator, FormatStrFormatter
from matplotlib import cm
import pandas as pd
from sklearn.metrics import silhouette_score
import scipy
from sklearn.cluster import KMeans
import pymc3 as pm
from theano import tensor as tt
import theano
eps = 1e-16


def genCov(dim):
    a = 2
    A = np.matrix([np.random.randn(dim) + np.random.randn(1)*a for i in range(dim)])
    A = A*np.transpose(A)
    D_half = np.diag(np.diag(A)**(-0.5))
    C = D_half*A*D_half
    sigma = np.diag(np.random.uniform(low=0,high=5,size=(dim)))
    return np.dot(np.dot(sigma,C),sigma)


#%%

#Number of features in our GMM
dim=20
N=10000

#Number of clusters
n_clusters = np.random.choice(2)

#Sample random mixing proportions
mix_prop = np.random.dirichlet([1 for x in range(n_clusters)])

#Initialise random covariance matrices
covs = [genCov(dim) for x in range(n_clusters)]

#Initialise random mean vectors
means = [np.random.uniform(low=1,high=20,size=dim) for x in range(n_clusters)]

#Sample cluster assignments
gAssign = np.random.multinomial(10000, mix_prop)
assignments = np.repeat(range(n_clusters),gAssign)

#Generate Data
mix = np.vstack([sp.multivariate_normal.rvs(mean=m,cov=c,size=g) for m,c,g in zip(means,covs,gAssign)])


# Peform the kernel density estimate
x = mix[:,0]
y = mix[:,1]
xmin = np.min(x)
xmax = np.max(x)
ymin = np.min(y)
ymax = np.max(y)
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = sp.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

# Plot the surface.
fig = plt.figure(figsize=(25,10))
ax = fig.add_subplot(projection='3d')

surf = ax.plot_surface(xx, yy, f, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
mate = fig.colorbar(surf, shrink=0.5, aspect=1)
mate.ax.set_xlabel('Kernel Density',fontsize=15)
plt.xlabel('Feature One',fontsize=15)
plt.ylabel('Feature Two',fontsize=15)
plt.title('Kernel Density Plot of GMM Data',fontsize=25)

plt.show()

#%%

DPGMM = mixture.BayesianGaussianMixture(n_components=15, 
                                                max_iter=1000,
                                                n_init=10,
                                                tol=1e-5, 
                                                init_params='kmeans', 
                                                weight_concentration_prior_type='dirichlet_process',
                                                weight_concentration_prior=1/10)
DPGMM.fit(mix)

#%%

fig, ax = plt.subplots(figsize=(8, 6))

plot_w = np.arange(15) + 1
ax.bar(plot_w - 0.5, np.sort(DPGMM.weights_)[::-1], width=1., lw=0)

ax.set_xlim(0.5, 15)
ax.set_xlabel('Component')
ax.set_ylabel('Posterior expected mixture weight')

#%%

n_to_select = 8
w = (-DPGMM.weights_).argsort()[:n_to_select]
mat = []
for ind2 in w:
    p1 = sp.multivariate_normal.pdf(mix,mean=DPGMM.means_[ind2],cov=DPGMM.covariances_[ind2]) + eps
    mat.append(p1)
clusts = pd.DataFrame(mat).T.idxmax(axis=1)

#%%

from sklearn.decomposition import PCA
pca= PCA(n_components=2)
X = pca.fit_transform(mix) # plotting with the PCA coordinates (why)

fig, ax = plt.subplots()

scatter = ax.scatter(X[:, 0], X[:, 1], c=assignments, cmap=plt.cm.viridis, alpha=.25);
plt.title("True label with %s clusters" %len(np.unique(assignments)));
plt.xlabel('PCA1')
plt.ylabel('PCA2')

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="Classes")
ax.add_artist(legend1)

#%%

fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=clusts, cmap=plt.cm.viridis, alpha=.25);
plt.title("DPGMM Predicted Labels with %s clusters" %len(np.unique(clusts)));
plt.xlabel('PCA1')
plt.ylabel('PCA2')

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="Classes")
ax.add_artist(legend1)
# %%


