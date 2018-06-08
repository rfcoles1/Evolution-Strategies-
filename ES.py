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
    eps = np.random.randn(config.num_policies, network.total_num)
    
    #evalate each policy over one episode
    for policy in range(config.num_policies):

        w_in_new = network.w_in + config.sigma*eps[policy,:network.num_in].reshape(network.w_in.shape)
        
        w_h_new = []
        curr = network.num_in
        for i in range(config.num_layers):        
            w_new = network.weights[i] + config.sigma*eps[policy, curr:curr+network.num_weights[i]].reshape(network.weights[i].shape)  
            w_h_new.append(w_new)
            curr += network.num_weights[i]

        w_out_new = network.w_out + config.sigma*eps[policy,network.total_num - network.num_out:].reshape(network.w_out.shape)
        
        #initial state
        s = env.reset()
        Reward[policy] = 0

        while True:
            #perform action based on this policy
            a = network.predict(s, w_in_new, w_out_new, w_h_new)
            a = np.argmax(a)
            s1, reward, done, _ = env.step(a)

            #collect reward
            Reward[policy] += reward
            s = s1

            if done:
                break
    std = np.std(Reward)
    if std == 0:
        continue
    F = (Reward - np.mean(Reward))/ np.std(Reward) #standardize the rewards to have a gaussian distribution
    weights_update = config.lr/(config.num_policies*config.sigma)*np.dot(eps.T,F) #update the weights using stochastic gradient estimate
   
    #update network based on reward achieved 
    network.w_in += weights_update[:network.num_in].reshape(network.w_in.shape)
    curr = network.num_in
    for i in range(config.num_layers):
        network.weights[i] += weights_update[curr:curr+network.num_weights[i]].reshape(network.weights[i].shape)
    network.w_out += weights_update[network.total_num - network.num_out:].reshape(network.w_out.shape)

    print episode, np.mean(Reward), np.max(Reward)
