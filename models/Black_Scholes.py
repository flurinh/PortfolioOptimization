import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

# Trajectories of the Black scholes model
# Let it run to initialize the following parameters, the trajectories
# are not needed afterwards


class BlackScholes:
    def __init__(self,
                 S=None,
                 N=30,  # time disrectization
                 W0=1,  # initial wealth
                 T=1,  # maturity
                 sigma=0.2,  # volatility in Black Scholes
                 mu=0.1,
                 r=0.0,
                 gamma=0.0,
                 R=4,  # exponent number of Trajectories
                 ):
        R = 10 ** R
        self.S = S
        assert self.S is not None, print("Black Scholes model requires synthetic data S.")
        print("Initializing Black Scholes model... {} trajectories are being evaluated!".format(int(R)))
        self.N = N
        logX = np.zeros((N, R))
        logX[0,] = np.log(W0) * np.ones((1, R))
        alpha = (mu - r) / (sigma ** 2 * (1 - gamma))
        for i in trange(R):
            for j in range(N - 1):
                increment = np.random.normal(mu * T / N, sigma * np.sqrt(T) / np.sqrt(N))
                logX[j + 1, i] = logX[j, i] + increment * alpha + r * T / N * (
                        1 - alpha) - alpha ** 2 * sigma ** 2 * T / (
                                         2 * N)
        self.X = np.exp(logX)
        self.loss = np.mean(-np.log(self.X[N - 1, :]))  # np.mean(-(W[N-1,:]**gamma))

    def plot_x(self, idx=0):
        plt.hist(self.X[self.N - 1, :])
        plt.show()

    def get_prediction(self):
        return self.X

    def get_loss(self):
        print("Mean of prediction loss:", np.mean(self.loss))
        print("Standard deviation of prediction loss:", np.std(self.loss))
        return self.loss
