import numpy as np
import gym

from ES_Config import Config
from ES_Network import Network

config = Config()
network = Network(config)

env = gym.make(Config.env_name)
Reward = np.zeros(config.num_policies)
sigma = config.min_sigma

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

                #collect reward
                #when doing classic control
                Reward[policy] += reward
                s = s1
    
                if done:
                    #when doing chemgym
                    #Reward[policy] += reward
                    break
    Reward /= config.num_iterations
    print episode, np.mean(Reward), np.max(Reward)
    if (episode % config.checkpoint_freq == 0) and (episode != 0):
        #find best policy to see if game is solved
        champ_ind = np.argmax(Reward)
        
        w_in_ch = network.w_in + sigma*eps[champ_ind,:network.num_in].reshape(network.w_in.shape)
        w_h_ch = []
        curr = network.num_in
        for i in range(config.num_layers):        
            w_ch = network.weights[i] + sigma*eps[champ_ind, curr:curr+network.num_weights[i]].reshape(network.weights[i].shape)  
            w_h_ch.append(w_ch)
            curr += network.num_weights[i]

        w_out_ch = network.w_out + sigma*eps[champ_ind,network.total_num - network.num_out:].reshape(network.w_out.shape)
        

        summed_reward= []
        for i in range(config.episodes_to_solve):
            s = env.reset()
            this_reward = 0
            while True:
                #perform action based on this policy
                a = network.predict(s, w_in_ch, w_out_ch, w_h_ch)
                a = np.argmax(a)
                s1, reward, done, _ = env.step(a)

                #collect reward
                #when doing classic control
                this_reward += reward
                s = s1
    
                if done:
                    #when doing chemgym
                    summed_reward.append(this_reward)
                    
                    break
        
        summed_reward = sum(summed_reward)
        score = summed_reward/config.episodes_to_solve            
        print 'Average score over ' + \
            str(config.episodes_to_solve) + ' episodes: ' + str(score) 
        if (score > config.score_to_solve):
            print 'The game is solve!'
            np.savez(config.model_path + str(episode) + '.npz',\
                w_in = w_in_ch, w_h = w_h_ch, w_out = w_out_ch)
            break

    std = np.std(Reward)
    if std == 0:
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
        network.w_out += weights_update[network.total_num - network.num_out:].reshape(network.w_out.shape)
        sigma = 0.1

    if (episode % config.checkpoint_freq == 0) and (episode != 0):
        print 'Saved Model'
        np.savez(config.model_path + str(episode) + '.npz',\
            w_in = network.w_in, w_h = network.weights, w_out = network.w_out)

