import os 
import sys 
import numpy as np
import gym

from ES_Config import Config
from ES_Network import Network

config = Config()
network = Network(config)

env = gym.make(Config.env_name)
Reward = np.zeros(config.num_policies)
sigma = config.min_sigma

#option to load a network
if config.load_model == True:
    if not os.path.exists(config.model_path + config.version_to_load):
        print 'Model does not exist'
    else:
        weights = np.load(config.model_path + config.version_to_load)
        
        network.w_in = weights['w_in']
        for i in range(config.num_layers):
            network.weights[i] = weights['w_h'][i]
        network.w_out = weights['w_out']

        print 'Model loaded'

for episode in range(config.num_episodes):
    #generate random variations around original policy
    eps = np.random.randn(config.num_policies, network.total_num)
    
    #evalate each policy over one episode
    for policy in range(config.num_policies):

        #create new weights by adding random noise to the existing 
        w_in_new = network.w_in + sigma*eps[policy,:network.num_in].reshape(network.w_in.shape)
        
        w_h_new = []
        curr = network.num_in
        for i in range(config.num_layers):        
            w_new = network.weights[i] + sigma*eps[policy, curr:curr+network.num_weights[i]].reshape(network.weights[i].shape)  
            w_h_new.append(w_new)
            curr += network.num_weights[i]

        w_out_new = network.w_out + sigma*eps[policy,network.total_num - network.num_out:].reshape(network.w_out.shape)
        
        #evaluate this policy
        Reward[policy] = 0
        for i in range(config.num_iterations):
            Reward[policy] += network.playthrough(env, w_in_new, w_out_new, w_h_new)

    Reward /= config.num_iterations
    print episode, np.mean(Reward), np.max(Reward)

    
    if (episode % config.checkpoint_freq == 0) and (episode != 0):
        #find best policy to see if game is solved
        champ_ind = np.argmax(Reward)
    
        #recreate the champion network
        w_in_ch = network.w_in + sigma*eps[champ_ind,:network.num_in].reshape(network.w_in.shape)
        
        w_h_ch = []
        curr = network.num_in
        for i in range(config.num_layers):        
            w_ch = network.weights[i] + sigma*eps[champ_ind, curr:curr+network.num_weights[i]].reshape(network.weights[i].shape)  
            w_h_ch.append(w_ch)
            curr += network.num_weights[i]

        w_out_ch = network.w_out + sigma*eps[champ_ind,network.total_num - network.num_out:].reshape(network.w_out.shape)
        
        #see if this network solves the game
        summed_reward = 0
        for i in range(config.episodes_to_solve):
            summed_reward += network.playthrough(env, w_in_ch, w_out_ch, w_h_ch)

        score = summed_reward/config.episodes_to_solve            
        print 'Average score over ' + \
            str(config.episodes_to_solve) + ' episodes: ' + str(score) 
        #if a network beats the game, that network is saved
        if (score >= config.score_to_solve):
            print 'The game is solve!'
            np.savez(config.model_path + str(episode) + '.npz',\
                w_in = w_in_ch, w_h = w_h_ch, w_out = w_out_ch)
            break

        sys.stdout.flush()

    #if not solved, update the network 
    std = np.std(Reward)
    if std == 0: #special case where every member of the population returns the same score
        #in this case, the range of exploration is increased
        if sigma < config.max_sigma:
            sigma += 0.1
    else:   
        A = (Reward - np.mean(Reward))/ np.std(Reward) #standardize the rewards to have a gaussian distribution
        #update the weights using stochastic gradient estimate
        weights_update = config.lr/(config.num_policies*sigma)*np.dot(eps.T,A)
   
        #update network based on reward achieved 
        network.w_in += weights_update[:network.num_in].reshape(network.w_in.shape)
        
        curr = network.num_in
        for i in range(config.num_layers):
            network.weights[i] += weights_update[curr:curr+network.num_weights[i]].reshape(network.weights[i].shape)
            curr += network.num_weights[i]
        network.w_out += weights_update[network.total_num - network.num_out:].reshape(network.w_out.shape)
        
        sigma = config.min_sigma #ensure the exploration is reset
    
    #updated network is saved
    if (episode % config.checkpoint_freq == 0) and (episode != 0):
        print 'Saved Model'
        np.savez(config.model_path + str(episode) + '.npz',\
            #w_in = network.w_in, w_h = network.weights, w_out = network.w_out)
            w_in = w_in_ch, w_h = w_h_ch, w_out = w_out_ch)
