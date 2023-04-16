import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
import torch
from reinforce import *

def main():
    env = gridworld.GridWorld(hard_version=False)
    n_episodes = int(2e3)
    optim = 'SGD'
    #optimizer = torch.optim.SGD([theta], lr = 1e-2)

    theta, reward_array_1 = reinforce(env, optim, n_episodes, gamma=1)

    fig, ax = plt.subplots(1,1)

    plt.plot(reward_array_1)
    plt.title("Reward - Reinforce with SGD - Easy Version")
    plt.xlabel("Number of epochs")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig('figures/reward_sgd_easy.png')

    plot_policy(theta, env, 'Reinforce with SGD - Easy Version' , 'policy_sgd_easy')

    optim = 'Adam'

    theta, reward_array_2 = reinforce(env, optim, n_episodes, gamma=1)

    fig, ax = plt.subplots(1,1)

    plt.plot(reward_array_2, label='Reinforce with Adam - Easy Version')
    plt.plot(reward_array_2, label='Reinforce with SGD - Easy Version')
    plt.title("Reward")
    plt.xlabel("Number of epochs")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig('figures/reward_adam_easy.png')

    plot_policy(theta, env, 'Reinforce with Adam - Easy Version' , 'policy_adam_easy')

    env = gridworld.GridWorld(hard_version=True)
    n_episodes = int(2e3)
    optim = 'SGD'
    #optimizer = torch.optim.SGD([theta], lr = 1e-2)

    theta, reward_array_3 = reinforce(env, optim, n_episodes, gamma=1)

    fig, ax = plt.subplots(1,1)
    plt.plot(reward_array_3)
    plt.title("Reward - Reinforce with SGD - Hard Version")
    plt.xlabel("Number of epochs")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig('figures/reward_sgd_hard.png')

    plot_policy(theta, env, 'Reinforce with SGD - Hard Version' , 'policy_sgd_hard')

    optim = 'Adam'

    theta, reward_array_4 = reinforce(env, optim, n_episodes, gamma=1)

    fig, ax = plt.subplots(1,1)
    plt.plot(reward_array_4, label='Reinforce with Adam - Hard Version')
    plt.plot(reward_array_3, label='Reinforce with SGD - Hard Version')
    plt.title("Reward")
    plt.xlabel("Number of epochs")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig('figures/reward_adam_hard.png')

    plot_policy(theta, env, 'Reinforce with Adam - Easy Version' , 'policy_adam_hard')


if __name__ == '__main__':
    main()
