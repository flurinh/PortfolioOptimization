from Models import *

# Todo: Here you can put your optimization class
# I suggest we use a wrapper class Optimizer to handle different classes of MODELS
# Here we implement a set of functions, either setting training properties or exectuing them, fit(), predict()


class Optimizer:
    def __init__(self,
                 training_id=0,
                 data=None,
                 model_id=0,
                 loss_id=0,
                 train=True,
                 evaluate=False):
        self.data = data
        print("Test 1, 2 ,3")
        assert self.data is not None, print("You should pass this optimizer a portfolio it can optimize, i.e. the data.")
        # next we chose a model
        if model_id == 0:
            self.model = EmptyModel()
        elif model_id == 1:
            self.model = "Some other model you can code"

    def select_loss(self, loss_id):
        # depending on what we want to optimize for... transaction cost etc
        if loss_id == 0:
            self.loss = 'Depends on whether we use pytorch or keras'
        elif loss_id == 1:
            self.loss = 'Depends on whether we use pytorch or keras'

    def fit(self):
        # Todo: create batches, set optimizer, set loss, train on batches
        pass

    def eval(self):
        # Todo: predict either on an evaluation dataset or real-time data
        pass

