from models.DefaultModel import *


# Todo: Here you can put your optimization class
# I suggest we use a wrapper class Optimizer to handle different classes of MODELS
# Here we implement a set of functions, either setting training properties or exectuing them, fit(), predict()


def custom_loss1(y_true, y_pred):
    z = - K.log(y_pred[:, 0])  # -((y_pred[:,0]**gamma-1)/gamma
    z = K.mean(z)
    return z


def custom_loss2(y_true, y_pred):
    ra = 0.1
    z = K.exp(- y_pred[:, 0] * ra)  # what is ra???
    z = K.mean(z)
    return z


class Optimizer:
    def __init__(self,
                 R = 4,  # exponent number of Trajectories
                 training_id=0,  # Used to differentiate different configurations and safe/load them
                 model_id=0,  # model-type (e.g. default-model from lecture -> model_id=0)
                 train=True,  # whether to train the model
                 evaluate=False,
                 ):
        self.data = {}
        self.initial_price = 0
        self.set_initial_price()
        self.initial_wealth = 0
        self.set_initial_wealth()
        # We initialize a model
        if model_id == 0:
            self.DM = DefaultModel()
            self.model = self.DM.get_model()
            print("using Default model\n", self.model)
            print("\n\n\n")
        elif model_id == 1:
            pass  # for another model, just use a new model_id
        self.params = self.DM.get_params()
        self.set_data(k=10 ** R)
        # Training
        if train:
            # self.train(0, 0.01, 'Adam', 100)
            pass
        else:
            # Todo: Load pretrained model
            pass
        # Evaluation
        if evaluate:
            pass

    def set_data(self, k):
        print("Generating {} data points!".format(k))
        xtrain = ([self.initial_price * np.ones((k, self.params['m']))] +
                  [np.zeros((k, self.params['m']))] +
                  [1 * np.ones((k, self.params['m']))] +
                  [self.initial_wealth * np.ones((k, self.params['m']))] +
                  [np.random.normal(self.params['mu'] * self.params['T'] / self.params['N'],
                                    self.params['sigma'] * np.sqrt(self.params['T']) / np.sqrt(self.params['N']),
                                    (k, self.params['m'])) for i in range(self.params['N'])])

        ytrain = np.zeros((k, 1 + self.params['N']))
        self.data.update({'Input': xtrain, 'Output': ytrain})

    def check_data(self):
        return self.data

    def set_initial_price(self, price=0):
        print("Setting initial price to {}.".format(price))
        self.initial_price = price

    def set_initial_wealth(self, wealth=0):
        print("Setting initial wealth to {}.".format(wealth))
        self.initial_wealth = wealth

    def select_loss(self, loss_id):
        # depending on what we want to optimize for... transaction cost etc
        if loss_id == 0:
            self.loss = custom_loss1
        elif loss_id == 1:
            self.loss = custom_loss2

    def set_optimizer(self, lr, opt):
        if opt is 'Adam':
            self.opt = optimizers.Adam(lr=lr)
        elif opt is 'AnotherOptimizer':
            # put whatever optimizer you want here
            pass

    def compile(self):
        print("Compiling model")
        self.model.compile(optimizer=self.opt, loss=self.loss)

    def train(self, loss_id=0, lr=0.01, opt='Adam', n_epochs=10, batch_size=64):
        self.select_loss(loss_id=loss_id)
        self.set_optimizer(lr=lr, opt=opt)
        self.compile()
        self.model.fit(x=self.data['Input'], y=self.data['Output'], epochs=n_epochs, verbose=True, \
                       batch_size=batch_size)

    def eval(self):
        # predict either on an evaluation dataset or real-time data
        y_pred = self.model.predict(self.data['Input'])
        print(np.mean(-np.log(y_pred[:, 0])))

        plt.hist(self.model.predict(self.data['Input'])[:, 0])
        plt.show()
        print(np.mean(self.model.predict(self.data['Input'])[:, 0]))
        print(np.std(self.model.predict(self.data['Input'])[:, 0]))

    def comp_alpha(self):
        """
        k = 10  # Choose a number between 1 and N-1
        Ktest = 60
        xtest = ([initialprice * np.ones((Ktest, m))] +
                 [np.zeros((Ktest, m))] +
                 [np.linspace(0.7, 1.5, Ktest)] +  # change this if you go to higher dimensions
                 [initialwealth * np.ones((Ktest, m))] +
                 [np.random.normal(mu * T / N, sigma * np.sqrt(T) / np.sqrt(N), (Ktest, m)) for i in range(N)])

        # Comparison of learned and true alpha
        s = np.linspace(0.7, 1.5, Ktest)

        for k in range(1, N):
            truestrat = (mu - r) / (sigma ** 2 * (1 - gamma)) * np.ones(Ktest)
            learnedstrat = model_Merton.predict(xtest)[:, k]
            plt.plot(s, learnedstrat, s, truestrat)
        plt.show()
        print((mu - r) / (sigma ** 2 * (1 - gamma)))
        """
        pass
