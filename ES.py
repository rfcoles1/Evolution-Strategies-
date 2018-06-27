import numpy as np
import gym
import os 

from ES_Config import Config
from ES_Network import Network

config = Config()
network = Network(config)

env = gym.make(Config.env_name)
Reward = np.zeros(config.num_policies)
sigma = config.min_sigma

if config.load_model == True:
    if not os.path.exists(config.model_path + config.version_to_load):
        print 'Model does not exist'
    else:
        weights = np.load(config.model_path + config.version_to_load)  
        print 'Model loaded'

        network.w_in = weights['w_in']
        
        curr = network.num_in
        for i in range(config.num_layers):
            network.weights[i] = weights['w_h'][i]
        network.w_out = weights['w_out']
        

for episode in range(config.num_episodes):

    #generate random variations around original policy
    eps = np.random.randn(config.num_policies, network.total_num)
    
    #evalate each policy over one episode
    for policy in range(config.num_policies):

        w_in_new = network.w_in + sigma*eps[policy,:network.num_in].reshape(network.w_in.shape)
        
        w_h_new = []
        curr = network.num_in
        for i in range(config.num_layers):        
            w_new = network.weights[i] + sigma*eps[policy, curr:curr+network.num_weights[i]].reshape(network.weights[i].shape)  
            w_h_new.append(w_new)
            curr += network.num_weights[i]

        w_out_new = network.w_out + sigma*eps[policy,network.total_num - network.num_out:].reshape(network.w_out.shape)
        #initial state
        Reward[policy] = 0
        for i in range(config.num_iterations):
            s = env.reset()
            while True:
                #perform action based on this policy
                a = network.predict(s, w_in_new, w_out_new, w_h_new)
                a = np.argmax(a)
                s1, reward, done, _ = env.step(a)
                
                '''when doing classic control'''
                Reward[policy] += reward
                s = s1
    
                if done:
                    '''when doing chemgym'''
                    #Reward[policy] += reward
                    break
    
    Reward /= config.num_iterations
    print episode, np.mean(Reward), np.max(Reward)
    champ_ind = np.argmax(Reward)

    #change the network to the best policy from previous generation
    network.w_in += sigma*eps[champ_ind,:network.num_in].reshape(network.w_in.shape)
    
    curr = network.num_in
    for i in range(config.num_layers):        
        network.weights[i] += sigma*eps[champ_ind, curr:curr+network.num_weights[i]].reshape(network.weights[i].shape)  
        curr += network.num_weights[i]

    network.w_out += sigma*eps[champ_ind,network.total_num - network.num_out:].reshape(network.w_out.shape)
        
    #save and check if solved
    if (episode % config.checkpoint_freq == 0) and (episode != 0):
        summed_reward = 0
        for i in range(config.episodes_to_solve):
            s = env.reset()
            while True:
                #perform action based on this policy
                a = network.predict(s, network.w_in, network.w_out, network.weights)
                a = np.argmax(a)
                s1, reward, done, _ = env.step(a)

                '''when doing classic control'''
                summed_reward += reward
                s = s1
    
                if done:
                    '''when doing chemgym'''
                    #summed_reward += reward
                    break
        
        score = summed_reward/config.episodes_to_solve            
        print 'Average score over ' + \
            str(config.episodes_to_solve) + ' episodes: ' + str(score) 
        if (score > config.score_to_solve):
            print 'The game is solve!'
            np.savez(config.model_path + str(episode) + '.npz',\
                w_in = network.w_in, w_h = network.weights, w_out = network.w_out)
            break
        else:
            print 'Saved Model'
            np.savez(config.model_path + str(episode) + '.npz',\
                w_in = network.w_in, w_h = network.weights, w_out = network.w_out)

