from Optimization import *
from Synthetic import *
from models.Black_Scholes import *


# Generate synthetic data which we work with, r corresponds to the log of # of trajectories, i.e. r = 5 -> N = 10 ** 5

r = 3


synth = SyntheticData(R=r)

# Visualize synthetic data
k = 10  # number of trajectories we want to visualize
z = np.arange(10 ** r)
np.random.shuffle(z)
synth.plot_s(all=False, idx=[z[x] for x in range(k)])  # you can set all to True to plot all curves, or set specific \
                                                       # indices (this is just to get a feel for how the data looks)

# Building the Black Scholes Model
bs = BlackScholes(synth.get_data(), R=r)
bs.get_loss()
bs.plot_x()


POptim = Optimizer(R=r)
data = POptim.check_data()


"""
Q1: How did we get to this input shape? (1000, 34)
A1: We use 4 types of inputs: INPUT = PRICE + TRADE + TRADEEVAL + WEALTH, where price, trade and eval have shape (k, m,) and
   wealth has shape (k, 1,), in our case m (price dimension is also 1)
   But then we also add (k, m, N) :=  (k, 1, 30), this results in (1000, 34) := (k, 3*m + 1 + N)
   
Q2: HOw did we get to this ouput shape?
A2: Idk


Q3: Why does the loss only consist of the output-term... basically Y is meaningless.
A3: Idk


Q4: Why does the loss not decrease?
A4: Probably related to Q2, Q3, same answer: Idk


Q5: Where is all of this code from?
A5: https://nbviewer.jupyter.org/urls/people.math.ethz.ch/~jteichma/lecture_ml_web/deep_portfolio_optimization_keras_merton.ipynb
"""

POptim.train(0, 0.01, 'Adam', 100)

"""
# Using real data (Yahoo stock prices)
States()
data = States.return_states()
"""
