import random
import numpy as np
import matplotlib.pyplot as plt
import torch

def policy(s, theta):
    logits = torch.nn.functional.softmax(theta[s,:],dim=0)
    return torch.distributions.categorical.Categorical(logits=logits)

def action(s, theta):
    return policy(s,theta).sample().item()

def plot_policy(theta, env, title, save_name):
    n_s = env.num_states

    actions = np.zeros(n_s)
    optimal_p = np.array([0, 0, 2, 0, 2, 0, 1, 1, 2, 2, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1])

    for i in range(n_s):
        actions[i] = action(i,theta)

    fig, ax = plt.subplots(1,1)
    plt.plot(actions, 'ro', label=title)
    plt.plot(optimal_p,'go',label="Optimal")
    plt.title("Policy")
    plt.xlabel("States")
    plt.ylabel("Actions")
    plt.legend()
    plt.savefig('figures/'+save_name+'.png')

def generate_episode(env, theta):
    T = env.max_num_steps
    states = torch.zeros(T+1, dtype = int)
    actions = torch.zeros(T+1, dtype = int)
    rewards = torch.zeros(T+1)
    states[0] = env.reset()
    done = False
    t = 0
    actions[t] = action(states[t].item(),theta)
    while not done:
        t+=1
        (states[t], rewards[t], done) = env.step(actions[t-1])
        actions[t] = action(states[t].item(),theta)
    return states, actions, rewards

def loss(G, theta, t, s, a, gamma):
    return -(gamma**t)*G*policy(s,theta).log_prob(torch.tensor([a]))


def reinforce(env, optim, n_episodes, gamma=1):
    n_s = env.num_states
    n_a = env.num_actions
    T = env.max_num_steps
    theta_0 = torch.rand(n_s,n_a)
    theta = theta_0.requires_grad_(requires_grad=True)
    reward_array = torch.zeros(n_episodes)

    if optim == 'SGD':
        optimizer = torch.optim.SGD([theta], lr = 1e-2)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam([theta], lr = 1e-2)

    for n in range(n_episodes):
        if n%1000 == 0:
            lr = 5e-3
        states, actions, rewards = generate_episode(env, theta)
        for t in range(T):
            G = 0
            for k in range(t+1,T,1):
                G+= gamma**(k-t-1)*rewards[k]
            optimizer.zero_grad()
            loss_value = loss(G, theta, t, states[t], actions[t], gamma)
            loss_value.backward()
            optimizer.step()
        reward_array[n] = rewards.sum()
    return theta, reward_array
