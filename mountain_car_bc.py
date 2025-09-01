import gymnasium as gym
import argparse
import pygame
from base_op import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device('cpu')

def collect_human_demos(num_demos):
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
    env = gym.make("MountainCar-v0",render_mode='rgb_array') 
    demos = collect_demos(env, keys_to_action=mapping, num_demos=num_demos, noop=1)
    return demos

def torchify_demos(sas_pairs):
    states = []
    actions = []
    next_states = []
    for s, a, s2 in sas_pairs:
        # Gymnasium返回的是(obs, info)，只取obs
        if isinstance(s, tuple):
            s = s[0]
        if isinstance(s2, tuple):
            s2 = s2[0]

        states.append(np.array(s, dtype=np.float32))
        actions.append(a)
        next_states.append(np.array(s2, dtype=np.float32))

    obs_torch = torch.from_numpy(np.stack(states)).float().to(device)
    acs_torch = torch.from_numpy(np.array(actions)).long().to(device)
    obs2_torch = torch.from_numpy(np.stack(next_states)).float().to(device)

    return obs_torch, acs_torch, obs2_torch

def train_policy(obs, acs, nn_policy, num_train_iters):
    """Train the behavoir cloning policy.

    Args:
        obs: The observation
        acs: The actions
        nn_policy: The network of policy
        num_train_iters: The number of iterations
    """
    optimizer = Adam(nn_policy.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()  # 离散动作分类问题使用交叉熵

    nn_policy.train()
    for i in range(num_train_iters):
        optimizer.zero_grad()
        logits = nn_policy(obs)
        loss = criterion(logits, acs)
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print(f"Iteration {i+1}/{num_train_iters}, Loss: {loss.item():.4f}")


class PolicyNetwork(nn.Module):
    '''
        Neural network: 3 hidden layers, 128 neurons per layer, ReLU activation function
    '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)  # output actions: logits

        # Normalization, to stabilize the data distribution
        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(128)
        self.ln3 = nn.LayerNorm(64)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        logits = self.fc4(x)
        return logits


#evaluate learned policy
def evaluate_policy(pi, num_evals, human_render=True):
    if human_render:
        env = gym.make("MountainCar-v0",render_mode='human') 
    else:
        env = gym.make("MountainCar-v0") 

    policy_returns = []
    for i in range(num_evals):
        done = False
        total_reward = 0
        obs, _ = env.reset()
        while not done:
            #take the action that the network assigns the highest logit value to
            #Note that first we convert from numpy to tensor and then we get the value of the 
            #argmax using .item() and feed that into the environment
            obs_tensor = torch.from_numpy(np.array(obs)).float().unsqueeze(0)
            action = torch.argmax(pi(obs_tensor)).item()
            # print(action)
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += rew
        print("reward for evaluation", i, total_reward)
        policy_returns.append(total_reward)

    print("average policy return", np.mean(policy_returns))
    print("min policy return", np.min(policy_returns))
    print("max policy return", np.max(policy_returns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_demos', default = 1, type=int, help="number of human demonstrations to collect")
    parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")

    args = parser.parse_args()

    #collect human demos
    demos = collect_human_demos(args.num_demos)

    #process demos
    obs, acs, _ = torchify_demos(demos)

    #train policy
    pi = PolicyNetwork()
    train_policy(obs, acs, pi, args.num_bc_iters)

    #evaluate learned policy
    evaluate_policy(pi, args.num_evals)

