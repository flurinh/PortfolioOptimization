# This is where we can put new model architectures, by using your own if-statement for the model_id parameter
# in the initialization of the Optimizer class each of us can test and run his own model - but this requires all of us
# to use the same ML-framework (pytorch/keras/TF).

import numpy as np
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Concatenate, Dropout, Subtract, \
    Flatten, MaxPooling2D, Multiply, Lambda, Add, Dot
from tensorflow.keras.backend import constant
from tensorflow.keras import optimizers

# from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras import initializers
from tensorflow.keras.constraints import max_norm
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt


class DefaultModel:
    def __init__(self,
                 N=30,  # number of timesteps (discrete)
                 T=1,  # maturity
                 m=1,  # dimension of price
                 d=3,  # number of layers in strategy
                 n=32,  # nodes in the first but last layers
                 ):
        self.N = N
        self.T = T
        self.sigma, self.mu, self.r = None, None, None
        self.set_normal_params()

        self.m = m
        self.d = d
        self.n = n

        self.price = Input(shape=(m,))
        self.trade = Input(shape=(m,))
        self.tradeeval = Input(shape=(m,))
        self.wealth = Input(shape=(1,))

        self.layers = []
        self.get_layers()

        self.inputs = [self.price] + [self.trade] + [self.tradeeval] + [self.wealth]
        self.outputs = None

        self.model = self.build_model()
        # self.get_model()

    def set_normal_params(self, sigma=None, mu=None, r=None):
        if sigma is None:
            self.sigma = 0.2
        else:
            self.sigma = sigma
        if mu is None:
            self.mu = 0.1
        else:
            self.mu = mu
        if r is None:
            self.r = 0.1
        else:
            self.r = r

    def get_params(self):
        return {'sigma': self.sigma, 'mu': self.mu, 'r': self.r, 'm': self.m, 'd': self.d, 'n': self.n, 'T': self.T,
                'N': self.N}

    def get_layers(self):
        for j in range(self.N):
            for i in range(self.d):
                if i < self.d - 1:
                    nodes = self.n
                    layer = Dense(nodes, activation='tanh', trainable=True,
                                  kernel_initializer=initializers.RandomNormal(0, 0.5),
                                  # kernel_initializer='random_normal',
                                  bias_initializer=initializers.RandomNormal(0, 0.5),
                                  name=str(i) + str(j))
                else:
                    nodes = self.m
                    layer = Dense(nodes, activation='linear', trainable=True,
                                  kernel_initializer=initializers.RandomNormal(0, 0.5),
                                  # kernel_initializer='random_normal',
                                  bias_initializer=initializers.RandomNormal(0, 0.5),
                                  name=str(i) + str(j))
                self.layers = self.layers + [layer]

    def build_model(self):
        outputhelper = []
        for j in range(self.N):
            strategy = self.price
            strategyeval = self.tradeeval
            for k in range(self.d):
                strategy = self.layers[k + (j) * self.d](strategy)  # strategy at j is the alpha at j
                strategyeval = self.layers[k + (j) * self.d](strategyeval)
            incr = Input(shape=(self.m,))
            logprice = Lambda(lambda x: K.log(x))(self.price)
            logprice = Add()([logprice, incr])
            pricenew = Lambda(lambda x: K.exp(x))(logprice)
            self.price = pricenew
            logwealth = Lambda(lambda x: K.log(x))(self.wealth)
            logwealth = Lambda(lambda x: x + self.r * self.T / self.N)(logwealth)
            helper1 = Multiply()([strategy, incr])
            # helper1 = Lambda()(lambda x : K.sum(x,axis=1))([helper1])
            logwealth = Add()([logwealth, helper1])
            helper2 = Multiply()([strategy, strategy])
            # helper2 = Lambda()(lambda x : K.sum(x,axis=1))([helper1])
            helper3 = Lambda(lambda x: x * self.sigma ** 2 / 2 * self.T / self.N)(helper2)
            logwealth = Subtract()([logwealth, helper3])
            helper4 = Lambda(lambda x: x * self.r * self.T / self.N)(strategy)
            logwealth = Subtract()([logwealth, helper4])
            wealthnew = Lambda(lambda x: K.exp(x))(logwealth)  # creating the wealth at time j+1
            self.inputs = self.inputs + [incr]
            outputhelper = outputhelper + [strategyeval]  # here we collect the strategies
            self.wealth = wealthnew
        self.outputs = self.wealth
        randomendowment = Lambda(lambda x: -0.0 * (K.abs(x - 1.0) + x - 1.0))(self.price)
        self.outputs = Add()([self.wealth, randomendowment])
        self.outputs = [self.outputs] + outputhelper
        self.outputs = Concatenate()(self.outputs)
        # Now return the model
        return Model(inputs=self.inputs, outputs=self.outputs)

    def get_model(self):
        return self.model


"""
# Implementing the outcoming of trading via neural networks
# Inputs is the training set below, containing the price S0,
# again we record the trading strategies on separate input variables 'tradeeval' to read them out easily later
"""
