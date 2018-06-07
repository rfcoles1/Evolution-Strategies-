import numpy as np
import gym

from ES_Config import Config
from ES_Network import Network

config = Config()
network = Network(config)

env = gym.make(Config.env_name)
Reward = np.zeros(config.num_policies)

for episode in range(config.num_episodes):

    #generate random variations around original policy
    eps = np.random.randn(config.num_policies, network.num_w1 + network.num_w2)

    #evalate each policy over one episode
    for policy in range(config.num_policies):

        w1_new = network.w1 + config.sigma*eps[policy,:network.num_w1].reshape(network.w1.shape)
        w2_new = network.w2 + config.sigma*eps[policy,network.num_w1:].reshape(network.w2.shape)

        #initial state
        s = env.reset()
    
        Reward[policy] = 0
        while True:
            a = network.predict(s, w1_new, w2_new)
            a = np.argmax(a)

            s1, reward, done, _ = env.step(a)

            #collect reward
            Reward[policy] += reward
            s = s1

            if done:
                break

    F = (Reward - np.mean(Reward))/ np.std(Reward) #standardize the rewards to have a gaussian distribution
    weights_update = config.lr/(config.num_policies*config.sigma)*np.dot(eps.T,F)
    
    network.w1 += weights_update[:network.num_w1].reshape(network.w1.shape)
    network.w2 += weights_update[network.num_w1:].reshape(network.w2.shape)
    
    print np.mean(Reward), np.max(Reward)
