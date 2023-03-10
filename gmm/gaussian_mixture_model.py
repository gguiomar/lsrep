#%%
import numpy as np
import matplotlib.pyplot as plt

class GaussianMixtureModel:

    def __init__(self, n_components):
        self.K = n_components

    def fit(self, X, n_iters=100):
        """
        Performs EM inference on a mixture of Gaussians.

        :param X: Data with shape (N, D).
        :return:  None.
        """
        μs, Σs, πs = self.init_params(X, self.K)
        log_likes = []

        for i in range(n_iters):
            resp = self.e_step(X, self.K, μs, Σs, πs)
            μs, Σs, πs = self.m_step(X, self.K, resp)
            self.means, self.covariances, self.weights = μs, Σs, πs
            log_likes.append(self.log_likelihood(X, μs, Σs, πs))

    def predict(self, X):
        """
        :param X: Observed data with shape (N, D).
        :return:  An N-vector with a mixture assignment to each observation.
        """
        assignments = np.zeros(X.shape[0])
        for n, x_n in enumerate(X):
            max_proba = 0
            assign_n  = 0
            for k, μk, Σk, in zip(range(self.K), self.means, self.covariances):
                proba = self.gaussian(x_n, μk, Σk)
                if proba > max_proba:
                    max_proba = proba
                    assign_n = k
            assignments[n] = assign_n
        return assignments

    def e_step(self, X, K, μs, Σs, πs):
        """
        Bishop eq 9.23.

        :param X:  Observed data with shape (N, D).
        :param K:  Number of mixture components.
        :param μs: Means with shape (K, D).
        :param Σs: Covariances with shape (K, D, D).
        :param πs: Mixture weights with shape (K).

        :return:   Responsibilities.
        """
        N, D = X.shape
        resp = np.empty((N, K))
        for n, x_n in enumerate(X):
            normalizer = self.xn_likelihood(x_n, μs, Σs, πs)
            for k in range(K):
                numer = πs[k] * self.gaussian(x_n, μs[k], Σs[k])
                resp[n, k] = numer / normalizer
        # Each row in the responsibilities matrix should be normalized.
        #assert np.isclose(resp.sum(axis=1), 1).all()
        return resp

    def m_step(self, X, K, resp):
        """
        Bishop eqs. 9.24, 9.25, 9.26.
        """
        N, D = X.shape
        μs_new = np.empty((K, D))
        Σs_new = np.empty((K, D, D))
        πs_new = np.empty(K)

        for k in range(K):
            # Bishop eq 9.27.
            Nk        = resp[:, k].sum()
            μs_new[k] = self.update_μk(X, resp[:, k], Nk)
            Σs_new[k] = self.update_Σk(X, resp[:, k], Nk, μs_new[k])
            # Bishop eq 9.26.
            πs_new[k] = Nk / N

        # πs_new should be a probability distribution.
        assert np.isclose(πs_new.sum(), 1)
        return μs_new, Σs_new, πs_new

    def init_params(self, X, K):
        """
        Initialize the means μ_k, covariances Σ_k and mixing coefficients π_k,
        and evaluate the initial value of the log likelihood.
        :return:
        """
        _, D  = X.shape

        μs = np.random.randn(K, D)

        # Use np.cov to ensure matrices are positive semi-definite.
        Σs = np.array([np.cov(X.T) for _ in range(K)])

        α = np.ones(K)
        πs = np.random.dirichlet(α)
        assert np.isclose(πs.sum(), 1)

        return μs, Σs, πs

    def update_μk(self, X, resp_k, Nk):
        """
        Bishop eq 9.24.
        """
        sum_ = 0
        for n, x_n in enumerate(X):
            sum_ += resp_k[n] * x_n
        return sum_ / Nk

    def update_Σk(self, X, resp_k, Nk, μk_new):
        """
        Bishop eq 9.25.
        """
        _, D = X.shape
        sum_ = 0
        for n, x_n in enumerate(X):
            tmp = x_n - μk_new
            sum_ += resp_k[n] * np.outer(tmp, tmp.T)
        Σk_new = sum_ / Nk
        assert Σk_new.shape == (D, D)
        return Σk_new
    
    def is_singular(self, x):
        """
        Check if a matrix is singular.
        """
        return np.linalg.matrix_rank(x) < x.shape[0] 

    def log_likelihood(self, X, μs, Σs, πs):
        """
        Bishop eq 9.14.

        :param X:  Observed data with shape (N, D).
        :param μs: Means with shape (K, D).
        :param Σs: Covariances with shape (K, D, D).
        :param πs: Mixture weights with shape (K).

        :return:   Log likelihood for all observations: ln p(X|π, μ, Σ).
        """
        #for e in Σs:
        #    print('singular', self.is_singular(e))
            
        total_ll = 0
        for x_n in X:
            l  = self.xn_likelihood(x_n, μs, Σs, πs)
            ll = np.log(l)
            total_ll += ll
        return total_ll

    def xn_likelihood(self, x_n, μs, Σs, πs):
        """
        Bishop eq 9.14 (n-th data point, no log).

        :param x_n: Single observed data point with shape (D).
        :param μs:  Means with shape (K, D).
        :param Σs:  Covariances with shape (K, D, D).
        :param πs:  Mixture weights with shape (K).

        :return:    Likelihood for the n-th observation.
        """
        l = 0
        for k in range(self.K):
            l += πs[k] * self.gaussian(x_n, μs[k], Σs[k])
        return l

    def gaussian(self, x_n, μ, Σ):
        """
        Bishop eq 1.52.

        :param x_n: An observation.
        :param μ:   A Gaussian mean.
        :param Σ:   A Gaussian variance.

        :return:    N(x|μ, Σ)
        """
        # The dimensionality of the data.
        D = x_n.shape[0]
        Σ_inv = np.linalg.inv(Σ)
        a = 1 / (2 * np.pi)**(D/2.)
        b = 1 / (np.linalg.det(Σ))**(1/2.)
        c = np.exp(-0.5 * np.dot(np.dot((x_n - μ).T, Σ_inv), x_n - μ))
        return a * b * c

    def sample(self, dim, n_samples):
        """
        Sample from a Gaussian mixture model.
        :param dim:
        :param n_samples:
        :return:
        """
        ks = np.random.choice(range(self.K), p=self.weights, size=n_samples)

        X_sim = np.zeros((n_samples, dim))
        Y_sim = np.zeros(n_samples)  # Cluster assignments.

        for i, k in enumerate(ks):
            μk, Σk = self.means[k], self.covariances[k]
            samp = np.random.multivariate_normal(μk, Σk)
            X_sim[i] = samp
            Y_sim[i] = k
        return X_sim, Y_sim

    def q(self, X):
        """
        :param Xi:
        :param Yi:
        :return:
        """
        N, D = X.shape
        πs = self.weights
        mixtures = np.zeros((self.K, N))
        for k in range(self.K):
            mix = np.zeros(N)
            for i, x in enumerate(X):
                μk, Σk = self.means[k], self.covariances[k]
                proba = πs[k] * self.gaussian(x, μk, Σk)
                mix[i] = proba
            mixtures[k] = mix
        assert len(mixtures) == self.K
        return np.array(mixtures).sum(axis=0)
 
# %%
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def is_singular(x):
    '''check if a matrix is singular'''
    return np.linalg.matrix_rank(x) < x.shape[0] 

def generate_pos_def_matrix(dim_x, dim_y):
    A = np.random.rand(dim_x, dim_y)
    return A.T.dot(A)

def generate_densities(n_densities, dim_x, dim_y): 
    densities = []
    sigma_list = [generate_pos_def_matrix(dim_x, dim_y) for _ in range(n_densities)]
    for i in range(n_densities):
        mu = np.random.rand(dim_x) + i
        sigma = sigma_list[i]
        weight = np.random.rand()
        densities.append({'mu': mu, 'sigma': sigma, 'weight': weight})
    pos_def = [is_pos_def(d['sigma']) for d in densities]
    singular = [is_singular(d['sigma']) for d in densities]
    return densities, pos_def, singular


def load_data_from_densities(densities, n_samples):
    k = 0
    samples = np.zeros((n_samples, 2))
    labels = np.zeros(n_samples)
    n_samples_class = n_samples // len(densities)
    print(samples.shape, labels.shape)
    for j, params in enumerate(densities):
        mu, sigma, weight = params['mu'], params['sigma'], params['weight']
        for i in range(n_samples_class):
            samples[k] = np.random.multivariate_normal(mu, sigma)
            labels[k] = j
            k += 1
    return samples, labels

def plot_mu_xy(densities):
    mu_x = [d['mu'][0] for d in densities]
    mu_y = [d['mu'][1] for d in densities]
    plt.scatter(mu_x, mu_y)
    plt.show()


def plot_data(X, Y, title):
    colors = ['r' if y == 0 else ('b' if y == 1 else 'g') for y in Y]
    plt.title(title)
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.show()
#%%
N_SAMPLES = 500
n_densities = 2
densities_gen, pos_def, singular = generate_densities(n_densities, 2, 2)
XX, YY = load_data_from_densities(densities_gen, N_SAMPLES)
plot_data(XX, YY, 'data')

gmm = GaussianMixtureModel(n_components = n_densities)
gmm.fit(XX)
X_sim, Y_sim = gmm.sample(XX.shape[1], 500)

plot_data(XX, YY, 'data')
plot_data(X_sim, Y_sim, 'sampled')

# %%

