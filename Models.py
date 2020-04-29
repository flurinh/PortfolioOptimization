# This is where we can put new model architectures, by using your own if-statement for the model_id parameter
# in the initialization of the Optimizer class each of us can test and run his own model - but this requires all of us
# to use the same ML-framework (pytorch/keras/TF).


class EmptyModel:
    def __init__(self):
        print("Here should be some form of network implementation, either keras, tensorflow or pytorch.")

    def forward(self, data_input):
        return data_input