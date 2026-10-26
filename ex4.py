import numpy as np
from typing import Callable

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

KERNEL_STRS = {
    'Laplacian': r'Laplacian, $\alpha={}$, $\beta={}$',
    'RBF': r'RBF, $\alpha={}$, $\beta={}$',
    'Spectral': r'Spectral, $\alpha={}$, $\beta={}$, $\gamma={}$',
    'NN': r'NN, $\alpha={}$, $\beta={}$'
}


def average_error(pred: np.ndarray, vals: np.ndarray):
    """
    Calculates the average squared error of the given predictions
    :param pred: the predicted values
    :param vals: the true values
    :return: the average squared error between the predictions and the true values
    """
    return np.mean((pred - vals)**2)


def RBF_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the RBF kernel
    :param alpha: the kernel variance
    :param beta: the kernel bandwidth
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        if x.ndim == 1: x = x[:, None]
        if y.ndim == 1: y = y[:, None]

        # Calculate squared Euclidean distance: (x-y)^2
        # Using broadcasting: (N, 1) - (1, M) -> (N, M)
        dist_sq = (x - y.T) ** 2
        return alpha * np.exp(-beta * dist_sq)
    return kern


def Laplacian_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the Laplacian kernel
    :param alpha: the kernel variance
    :param beta: the kernel bandwidth
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        if x.ndim == 1: x = x[:, None]
        if y.ndim == 1: y = y[:, None]

        # L1 distance: |x-y|
        dist = np.abs(x - y.T)
        return alpha * np.exp(-beta * dist)
    return kern


def Spectral_kernel(alpha: float, beta: float, gamma: float) -> Callable:
    """
    An implementation of the Spectral kernel (see https://arxiv.org/pdf/1302.4245.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        if x.ndim == 1: x = x[:, None]
        if y.ndim == 1: y = y[:, None]

        dist = x - y.T  # Signed distance
        dist_sq = dist ** 2

        return alpha * np.exp(-beta * dist_sq) * np.cos(np.pi * dist / gamma)
    return kern


def NN_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the Neural Network kernel (see section 4.2.3 of http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        if x.ndim == 1: x = x[:, None]
        if y.ndim == 1: y = y[:, None]

        # Assuming x and y are shape (N, 1) and (M, 1)
        # x_tilde dot y_tilde = 1 + x dot y
        x_dot_y = 1 + x @ y.T  # Shape (N, M)

        # Terms for denominator
        # 1 + 2*beta*(1 + x^2)
        x_norm_term = 1 + 2 * beta * (1 + x ** 2)  # Shape (N, 1)
        y_norm_term = 1 + 2 * beta * (1 + y ** 2)  # Shape (M, 1)

        numerator = 2 * beta * x_dot_y
        denominator = np.sqrt(x_norm_term @ y_norm_term.T)

        return alpha * (2.0 / np.pi) * np.arcsin(numerator / denominator)
    return kern


class GaussianProcess:

    def __init__(self, kernel: Callable, noise: float):
        """
        Initialize a GP model with the specified kernel and noise
        :param kernel: the kernel to use when fitting the data/predicting
        :param noise: the sample noise assumed to be added to the data
        """
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.L = None  # Cholesky factor of K + noise*I
        self.alpha = None  # L.T \ (L \ y)

    def fit(self, X, y) -> 'GaussianProcess':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        self.X_train = X
        self.y_train = y

        # Compute Kernel Matrix K
        K = self.kernel(X, X)

        # Add noise to diagonal
        K_y = K + self.noise * np.eye(len(X))

        # Cholesky Decomposition (more stable than inversion)
        # K_y = L @ L.T
        try:
            self.L = np.linalg.cholesky(K_y)
        except np.linalg.LinAlgError:
            # if matrix is not positive definite (numerical issues)
            # Add a small jitter
            self.L = np.linalg.cholesky(K_y + 1e-6 * np.eye(len(X)))

        # Compute alpha = (K + noise*I)^-1 * y
        # solve L @ L.T @ alpha = y
        #  solve L @ z = y
        z = np.linalg.solve(self.L, y)
        # solve L.T @ alpha = z
        self.alpha = np.linalg.solve(self.L.T, z)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the MMSE regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        # if not trained, return prior mean (assumed 0)
        if self.X_train is None:
            return np.zeros(X.shape[0])

        # K_star = k(X_test, X_train)
        K_star = self.kernel(X, self.X_train)

        # f_bar = K_star @ alpha
        return K_star @ self.alpha

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def sample(self, X) -> np.ndarray:
        """
        Sample a function from the posterior
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the sample (same shape as X)
        """
        # calculate mean
        mu = self.predict(X)

        # calculate full covariance
        K_ss = self.kernel(X, X)  # prior covariance

        if self.X_train is not None:
            # posterior covariance: K_ss - K_s @ (K + noise*I)^-1 @ K_s.T
            K_s = self.kernel(X, self.X_train)
            # v = L^-1 @ K_s.T
            v = np.linalg.solve(self.L, K_s.T)
            cov = K_ss - v.T @ v
        else:
            cov = K_ss

        # sample from multivariate normal
        # small jitter for numerical stability during sampling
        cov = cov + 1e-9 * np.eye(len(X))
        return np.random.multivariate_normal(mu, cov)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        # only need the diagonal of the covariance matrix for std
        K_ss_diag = np.diag(self.kernel(X, X))

        if self.X_train is None:
            return np.sqrt(K_ss_diag)

        # posterior Variance: diag(K_ss) - diag(K_s @ (K+noise)^-1 @ K_s.T)
        K_s = self.kernel(X, self.X_train)

        # v = L^-1 @ K_s.T
        v = np.linalg.solve(self.L, K_s.T)

        # subtracted term is sum(v^2, axis=0) == diag(v.T @ v)
        var_reduction = np.sum(v ** 2, axis=0)

        var = K_ss_diag - var_reduction

        # clip negative values due to numerical errors
        var = np.maximum(var, 0)

        return np.sqrt(var)

    def log_evidence(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the model's log-evidence under the training data
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the log-evidence of the model under the data points
        """
        self.fit(X, y)

        # Log Evidence = -0.5 * y.T @ alpha - sum(log(diag(L))) - 0.5 * N * log(2*pi)

        term1 = -0.5 * y @ self.alpha

        # det(K_y) = det(L @ L.T) = det(L)^2
        # log(det(K_y)) = 2 * log(det(L)) = 2 * sum(log(diag(L)))
        term2 = -np.sum(np.log(np.diag(self.L)))  # This is -0.5 * log|K_y|

        N = len(y)
        term3 = -0.5 * N * np.log(2 * np.pi)

        return term1 + term2 + term3


def main():
    # ------------------------------------------------------ section 2.1
    xx = np.linspace(-5, 5, 500)
    x, y = np.array([-2, -1, 0, 1, 2]), np.array([2.4, .9, 2.8, -2.9, -1.5])

    # ------------------------------ questions 2 and 3
    # choose kernel parameters
    params = [
        # Laplacian kernels
        ['Laplacian', Laplacian_kernel, 1, 0.25],
        ['Laplacian', Laplacian_kernel, 1, 1],
        ['Laplacian', Laplacian_kernel, 1, 5],

        # RBF kernels
        ['RBF', RBF_kernel, 1, 0.25],
        ['RBF', RBF_kernel, 1, 1],
        ['RBF', RBF_kernel, 1, 10],

        # Gibbs kernels (Spectral)
        ['Spectral', Spectral_kernel, 1, 0.1, 3],
        ['Spectral', Spectral_kernel, 1, 0.1, 1],
        ['Spectral', Spectral_kernel, 1, 0.01, 1],

        # Neural network kernels
        ['NN', NN_kernel, 1, 0.25],
        ['NN', NN_kernel, 1, 1.0],
        ['NN', NN_kernel, 1, 5.0],
    ]
    noise_var = 0.05

    # plot all of the chosen parameter settings
    for p in params:
        # create kernel according to parameters chosen
        k = p[1](*p[2:])  # p[1] is the kernel function while p[2:] are the kernel parameters

        # initialize GP with kernel defined above
        gp = GaussianProcess(k, noise_var)

        # ---------------- Prior Plot ----------------
        plt.figure(figsize=(10, 4))

        # 1. Plot mean (0) and confidence interval of Prior
        prior_mean = np.zeros_like(xx)
        prior_std = gp.predict_std(xx)

        plt.fill_between(xx, prior_mean - 2 * prior_std, prior_mean + 2 * prior_std, alpha=.2, color='blue',
                         label='Confidence')
        plt.plot(xx, prior_mean, 'b--', lw=2, label='Mean')

        # 2. Sample 5 functions from Prior
        for i in range(5):
            plt.plot(xx, gp.sample(xx), lw=1, alpha=0.6)

        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title(f'Prior: {KERNEL_STRS[p[0]].format(*p[2:])}')
        plt.ylim([-5, 5])
        plt.legend()
        plt.show()

        # ---------------- Posterior Plot ----------------
        # fit the GP to the data
        gp.fit(x, y)
        m = gp.predict(xx)
        s = gp.predict_std(xx)

        plt.figure(figsize=(10, 4))

        # Plot Posterior mean and confidence
        plt.fill_between(xx, m - 2 * s, m + 2 * s, alpha=.3, color='orange', label='Confidence')
        plt.plot(xx, m, 'r', lw=2, label='Mean')

        # Sample 5 functions from Posterior
        for i in range(5):
            plt.plot(xx, gp.sample(xx), lw=1, alpha=0.6)

        # Plot training data
        plt.scatter(x, y, 50, 'k', zorder=10, label='Data')

        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title(f'Posterior: {KERNEL_STRS[p[0]].format(*p[2:])}')
        plt.ylim([-5, 5])
        plt.legend()
        plt.show()

    # ------------------------------ question 4
    # define range of betas
    betas = np.linspace(0.1, 7, 101)
    noise_var = .27

    # calculate the evidence for each of the kernels
    evidence = [GaussianProcess(RBF_kernel(1, beta=b), noise_var).log_evidence(x, y) for b in betas]

    # plot the evidence as a function of beta
    plt.figure()
    plt.plot(betas, evidence, lw=2)
    plt.xlabel(r'$\beta$')
    plt.ylabel('log-evidence')
    plt.title('Log-Evidence vs Beta (RBF Kernel)')
    plt.grid(True)
    plt.show()

    # extract betas that had the min, median and max evidence
    srt = np.argsort(evidence)
    # min_ev: lowest evidence, max_ev: highest evidence
    min_ev = betas[srt[0]]
    median_ev = betas[srt[(len(evidence) - 1) // 2]]
    max_ev = betas[srt[-1]]

    print(f"Beta with Min Evidence: {min_ev}")
    print(f"Beta with Median Evidence: {median_ev}")
    print(f"Beta with Max Evidence: {max_ev}")

    # plot the mean of the posterior of a GP using the extracted betas on top of the data
    plt.figure(figsize=(10, 6))

    # Max Evidence
    gp_max = GaussianProcess(RBF_kernel(1, beta=max_ev), noise_var).fit(x, y)
    m_max, s_max = gp_max.predict(xx), gp_max.predict_std(xx)
    plt.plot(xx, m_max, 'g', lw=2, label=f'Max Evidence ($\\beta={max_ev:.2f}$)')
    plt.fill_between(xx, m_max - 2 * s_max, m_max + 2 * s_max, color='g', alpha=0.1)

    # Median Evidence
    gp_med = GaussianProcess(RBF_kernel(1, beta=median_ev), noise_var).fit(x, y)
    m_med, s_med = gp_med.predict(xx), gp_med.predict_std(xx)
    plt.plot(xx, m_med, 'b', lw=2, label=f'Median Evidence ($\\beta={median_ev:.2f}$)')
    plt.fill_between(xx, m_med - 2 * s_med, m_med + 2 * s_med, color='b', alpha=0.1)

    # Min Evidence
    gp_min = GaussianProcess(RBF_kernel(1, beta=min_ev), noise_var).fit(x, y)
    m_min, s_min = gp_min.predict(xx), gp_min.predict_std(xx)
    plt.plot(xx, m_min, 'r', lw=2, label=f'Min Evidence ($\\beta={min_ev:.2f}$)')
    plt.fill_between(xx, m_min - 2 * s_min, m_min + 2 * s_min, color='r', alpha=0.1)

    plt.scatter(x, y, 50, 'k', zorder=10, label='Data')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.title('Posterior Means for different Evidence levels')
    plt.legend()
    plt.ylim([-5, 5])
    plt.show()


if __name__ == '__main__':
    main()

#
# def main():
#     # ------------------------------------------------------ section 2.1
#     xx = np.linspace(-5, 5, 500)
#     x, y = np.array([-2, -1, 0, 1, 2]), np.array([2.4, .9, 2.8, -2.9, -1.5])
#
#     # ------------------------------ questions 2 and 3
#     # choose kernel parameters
#     params = [
#         # Laplacian kernels
#         ['Laplacian', Laplacian_kernel, 1, 0.25],           # insert your parameters, order: alpha, beta
#         ['Laplacian', Laplacian_kernel, None, None],        # insert your parameters, order: alpha, beta
#         ['Laplacian', Laplacian_kernel, None, None],        # insert your parameters, order: alpha, beta
#
#         # RBF kernels
#         ['RBF', RBF_kernel, 1, 0.25],                       # insert your parameters, order: alpha, beta
#         ['RBF', RBF_kernel, None, None],                    # insert your parameters, order: alpha, beta
#         ['RBF', RBF_kernel, None, None],                    # insert your parameters, order: alpha, beta
#
#         # Gibbs kernels
#         ['Spectral', Spectral_kernel, 1, .5, 3],            # insert your parameters, order: alpha, beta, gamma
#         ['Spectral', Spectral_kernel, None, None, None],    # insert your parameters, order: alpha, beta, gamma
#         ['Spectral', Spectral_kernel, None, None, None],    # insert your parameters, order: alpha, beta, gamma
#
#         # Neurel network kernels
#         ['NN', NN_kernel, 1, 0.25],                         # insert your parameters, order: alpha, beta
#         ['NN', NN_kernel, None, None],                      # insert your parameters, order: alpha, beta
#         ['NN', NN_kernel, None, None],                      # insert your parameters, order: alpha, beta
#     ]
#     noise_var = 0.05
#
#     # plot all of the chosen parameter settings
#     for p in params:
#         # create kernel according to parameters chosen
#         k = p[1](*p[2:])  # p[1] is the kernel function while p[2:] are the kernel parameters
#
#         # initialize GP with kernel defined above
#         gp = GaussianProcess(k, noise_var)
#
#         # ---------------- Prior Plot ----------------
#         plt.figure(figsize=(10, 4))
#
#         # 1. Plot mean (0) and confidence interval of Prior
#         prior_mean = np.zeros_like(xx)
#         prior_std = gp.predict_std(xx)
#
#         plt.fill_between(xx, prior_mean - 2 * prior_std, prior_mean + 2 * prior_std, alpha=.2, color='blue',
#                          label='Confidence')
#         plt.plot(xx, prior_mean, 'b--', lw=2, label='Mean')
#
#         # 2. Sample 5 functions from Prior
#         for i in range(5):
#             plt.plot(xx, gp.sample(xx), lw=1, alpha=0.6)
#
#         plt.xlabel('$x$')
#         plt.ylabel('$f(x)$')
#         plt.title(f'Prior: {KERNEL_STRS[p[0]].format(*p[2:])}')
#         plt.ylim([-5, 5])
#         plt.legend()
#         plt.show()
#
#         # ---------------- Posterior Plot ----------------
#         # fit the GP to the data
#         gp.fit(x, y)
#         m = gp.predict(xx)
#         s = gp.predict_std(xx)
#
#         plt.figure(figsize=(10, 4))
#
#         # 1. Plot Posterior mean and confidence
#         plt.fill_between(xx, m - 2 * s, m + 2 * s, alpha=.3, color='orange', label='Confidence')
#         plt.plot(xx, m, 'r', lw=2, label='Mean')
#
#         # 2. Sample 5 functions from Posterior
#         for i in range(5):
#             plt.plot(xx, gp.sample(xx), lw=1, alpha=0.6)
#
#         # 3. Plot training data
#         plt.scatter(x, y, 50, 'k', zorder=10, label='Data')
#
#         plt.xlabel('$x$')
#         plt.ylabel('$f(x)$')
#         plt.title(f'Posterior: {KERNEL_STRS[p[0]].format(*p[2:])}')
#         plt.ylim([-5, 5])
#         plt.legend()
#         plt.show()
#
#     # ------------------------------ question 4
#     # define range of betas
#     betas = np.linspace(0.1, 7, 101)
#     noise_var = .27
#
#     # calculate the evidence for each of the kernels
#     evidence = [GaussianProcess(RBF_kernel(1, beta=b), noise_var).log_evidence(x, y) for b in betas]
#
#     # plot the evidence as a function of beta
#     plt.figure()
#     plt.plot(betas, evidence, lw=2)
#     plt.xlabel(r'$\beta$')
#     plt.ylabel('log-evidence')
#     plt.title('Log-Evidence vs Beta (RBF Kernel)')
#     plt.grid(True)
#     plt.show()
#
#     # extract betas that had the min, median and max evidence
#     srt = np.argsort(evidence)
#     # min_ev: lowest evidence, max_ev: highest evidence
#     min_ev = betas[srt[0]]
#     median_ev = betas[srt[(len(evidence) - 1) // 2]]
#     max_ev = betas[srt[-1]]
#
#     print(f"Beta with Min Evidence: {min_ev}")
#     print(f"Beta with Median Evidence: {median_ev}")
#     print(f"Beta with Max Evidence: {max_ev}")
#
#     # plot the mean of the posterior of a GP using the extracted betas on top of the data
#     plt.figure(figsize=(10, 6))
#
#     # Max Evidence
#     gp_max = GaussianProcess(RBF_kernel(1, beta=max_ev), noise_var).fit(x, y)
#     m_max, s_max = gp_max.predict(xx), gp_max.predict_std(xx)
#     plt.plot(xx, m_max, 'g', lw=2, label=f'Max Evidence ($\\beta={max_ev:.2f}$)')
#     plt.fill_between(xx, m_max - 2 * s_max, m_max + 2 * s_max, color='g', alpha=0.1)
#
#     # Median Evidence
#     gp_med = GaussianProcess(RBF_kernel(1, beta=median_ev), noise_var).fit(x, y)
#     m_med, s_med = gp_med.predict(xx), gp_med.predict_std(xx)
#     plt.plot(xx, m_med, 'b', lw=2, label=f'Median Evidence ($\\beta={median_ev:.2f}$)')
#     plt.fill_between(xx, m_med - 2 * s_med, m_med + 2 * s_med, color='b', alpha=0.1)
#
#     # Min Evidence
#     gp_min = GaussianProcess(RBF_kernel(1, beta=min_ev), noise_var).fit(x, y)
#     m_min, s_min = gp_min.predict(xx), gp_min.predict_std(xx)
#     plt.plot(xx, m_min, 'r', lw=2, label=f'Min Evidence ($\\beta={min_ev:.2f}$)')
#     plt.fill_between(xx, m_min - 2 * s_min, m_min + 2 * s_min, color='r', alpha=0.1)
#
#     plt.scatter(x, y, 50, 'k', zorder=10, label='Data')
#     plt.xlabel(r'$x$')
#     plt.ylabel(r'$f(x)$')
#     plt.title('Posterior Means for different Evidence levels')
#     plt.legend()
#     plt.ylim([-5, 5])
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()
#
#
#
