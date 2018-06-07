import os
import gym
import multiprocessing
import time

class Config:

    env_name = 'Acrobot-v1' 
    if env_name == 'CartPole-v0' or env_name == 'MountainCar-v0' or env_name == 'Acrobot-v1':
        mode = 'discrete'
    elif env_name == 'MountainCarContinuous-v0' or env_name == 'Pendulum-v0': 
        mode = 'continuous'

    game = gym.make(env_name)
    s_size = len(game.reset()) 
    if mode == 'discrete':
        a_size = game.action_space.n
    else:
        a_size = game.action_space.shape[0]
        a_bounds = [game.action_space.low, game.action_space.high]
        a_range = game.action_space.high - game.action_space.low

    num_policies = 100 
    num_episodes = 5000 
    num_hidden = 128
    lr = 1e-4
    sigma = 0.1    
    

     

