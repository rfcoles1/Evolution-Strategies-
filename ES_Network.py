import numpy as np 

class Network():
    def __init__(self, config):
        
        self.w1 = np.random.randn(config.num_hidden,config.s_size)/np.sqrt(config.s_size)
        self.w2 = np.random.randn(config.a_size,config.num_hidden)/np.sqrt(config.num_hidden)
        self.num_w1 = len(self.w1.flatten())
        self.num_w2 = len(self.w2.flatten())

    def layer

    def predict(self, s, w1, w2):
        h = np.dot(w1,s) #input to hidden layer
        h[h<0] = 0 #relu
        out = np.dot(w2,h) #hidden layer to output
        out = 1.0/(1.0 + np.exp(-out)) #sigmoid
        return out
