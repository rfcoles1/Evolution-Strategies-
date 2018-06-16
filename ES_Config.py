import os
import gym
import multiprocessing
import time

"""
import sys
sys.path.insert(0,'../')
import ChemGym
"""

class Config:

    env_name = 'MountainCar-v0' 

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
    num_episodes = 10000
    checkpoint_freq = 1 #number of generations between checkpoints 
    num_iterations = 3 #number of games each policy is evaluated for, score is averaged
    
    num_layers = 2 
    num_hidden = 128
    lr = 0.05
    min_sigma = 0.1   
    max_sigma = 0.5
    
    model_path = './models/' + str(env_name) + '/' #+ '_' + str(time.ctime()) + '/' 
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    score_to_solve = -110
    episodes_to_solve = 100

    load_model = False
    
