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

class PerturbedMountainCarEnv(gym.Wrapper):
    def __init__(self, env, noise_std=0.05, perturb_prob=0.3):
        """
        The MountainCar environment with random perturbation
        Args:
            env: gym env
            noise_std: standard deviation of state observation noise
            perturb_prob: The probability of applying the perturbation by each step
        """
        super().__init__(env)
        self.noise_std = noise_std
        self.perturb_prob = perturb_prob
        self.original_step = self.env.step
        
    def step(self, action):

        if np.random.random() < self.perturb_prob:
            # choosing action randomly
            perturbed_action = np.random.choice([0, 1, 2])
            obs, reward, terminated, truncated, info = self.original_step(perturbed_action)
        else:
            obs, reward, terminated, truncated, info = self.original_step(action)
        
        # perturbation on observation
        # obs = obs + np.random.normal(0, self.noise_std, obs.shape)
            
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def create_env(perturbed=False, render_mode='rgb_array', max_steps=500, noise_std=0.05, perturb_prob=0.3):

    env = gym.make("MountainCar-v0", render_mode=render_mode, max_episode_steps=max_steps)
    
    if perturbed:
        env = PerturbedMountainCarEnv(
            env, 
            noise_std=noise_std, 
            perturb_prob=perturb_prob,
        )
    
    return env


def collect_human_demos(num_demos, env):
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
    demos = collect_demos(env, keys_to_action=mapping, num_demos=num_demos, noop=1)
    return demos

def torchify_demos(sas_pairs):
    states = []
    actions = []
    next_states = []
    for s, a, s2 in sas_pairs:
        # in some env, the return will be tuple of {obs, info}
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
    """Train the behavior cloning policy.

    Args:
        obs: The observation
        acs: The index of actions
        nn_policy: The network of policy
        num_train_iters: The number of iterations
    """
    optimizer = Adam(nn_policy.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()  # 离散动作分类问题使用交叉熵

    nn_policy.train()
    for i in range(num_train_iters):
        optimizer.zero_grad()
        logits = nn_policy(obs)
        # print(f"logits: {logits.shape}")
        # print(f"acs index: {acs.shape}")
        loss = criterion(logits, acs)
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print(f"Iteration {i+1}/{num_train_iters}, Loss: {loss.item():.4f}")


class PolicyNetwork(nn.Module):
    '''
        Neural network: 3 hidden layers, 128 neurons per layer, ReLU activation function
        input: (prev_obs, action, obs)
        output: probability of action
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

def noisy_obs(obs, noise_std=0.01):
    return obs + np.random.normal(0, noise_std, size=obs.shape)


#evaluate learned policy
def evaluate_policy(pi, num_evals, env):

    policy_returns = []
    for i in range(num_evals):
        done = False
        total_reward = 0
        obs, _ = env.reset()
        while not done:
            noisy_observation = noisy_obs(np.array(obs), noise_std=0.02)
            obs_tensor = torch.from_numpy(np.array(noisy_observation)).float().unsqueeze(0)
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
    parser.add_argument('--enable_perturb', default=False, type=bool, help="if enable perturbation for action and observation")
    parser.add_argument('--env_max_step', default=500, type=int, help="max step limitation to env")

    args = parser.parse_args()
    env = create_env(perturbed=args.enable_perturb, max_steps = args.env_max_step)
    #collect human demos
    demos = collect_human_demos(args.num_demos, env)

    #process demos
    obs, acs, _ = torchify_demos(demos)

    #train policy
    pi = PolicyNetwork()
    train_policy(obs, acs, pi, args.num_bc_iters)

    env = create_env(perturbed=args.enable_perturb, max_steps = args.env_max_step, render_mode='human')
    #evaluate learned policy
    evaluate_policy(pi, args.num_evals, env)

