import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

# Trajectories of the Black scholes model
# Let it run to initialize the following parameters, the trajectories
# are not needed afterwards


class SyntheticData:
    def __init__(self,
                 N=30,  # time disrectization
                 S0=1,  # initial value of the asset
                 T=1,  # maturity
                 sigma=0.2,  # volatility in Black Scholes
                 mu=0.1,
                 R=4,  # exponent number of Trajectories
                 ):
        R = 10 ** R
        print("Initializing synthetic data S... {} trajectories are being generated!".format(int(R)))
        logS = np.zeros((N, R))
        logS[0,] = np.log(S0) * np.ones((1, R))

        for i in trange(R):
            for j in range(N - 1):
                increment = np.random.normal(mu * T / N - (sigma) ** 2 * T / (2 * N), sigma * np.sqrt(T) / np.sqrt(N))
                logS[j + 1, i] = logS[j, i] + increment
        self.S = np.exp(logS)

    def plot_s(self, all=False, idx=0):
        if all:
            plt.plot(self.S[:, :])
            plt.show()
        else:
            plt.plot(self.S[:, idx])
            plt.show()

    def get_data(self):
        return self.S
