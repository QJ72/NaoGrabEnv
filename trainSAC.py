import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from typing import Union, Optional

from tqdm import trange

import torch
import torch.optim as optim

import gymnasium
import numpy as np

from RLAlg.alg.sac import SAC
from RLAlg.buffer.replay_buffer import ReplayBuffer
from RLAlg.nn.steps import StochasticContinuousPolicyStep, DiscretePolicyStep, ValueStep
from model.modelSAC import Actor, Critic

import register_env

from env.nao_grab_env_cfg import NaoGrabEnvCfg

def process_obs(obs):
    features = obs["policy"]

    return features

class Trainer:
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.envs = gymnasium.make("NaoGrabEnv-v0", cfg=NaoGrabEnvCfg())

        self.env_nums, self.obs_dim = self.envs.observation_space.shape

        self.steps = 50
        self.buffer_steps = 1000
        self.rollout_steps = self.steps

        obs_space = self.envs.observation_space.shape
        action_space = self.envs.action_space.shape

        self.action_dim = action_space[-1]

        self.actor = Actor(self.obs_dim, self.action_dim, [128, 128]).to(self.device)
        self.critic = Critic(self.obs_dim, self.action_dim, [128, 128]).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.action_dim, [128, 128]).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.replay_buffer = ReplayBuffer(self.env_nums, self.buffer_steps, device=self.device)
        self.replay_buffer.create_storage_space("observations", (self.obs_dim, ), torch.float32)
        self.replay_buffer.create_storage_space("next_observations", (self.obs_dim, ), torch.float32)
        self.replay_buffer.create_storage_space("actions", (self.action_dim, ), torch.float32)
        self.replay_buffer.create_storage_space("rewards", (), torch.float32)
        self.replay_buffer.create_storage_space("dones", (), torch.float32)

        self.batch_keys = ["observations", "next_observations", "actions", "rewards", "dones"]
        
        self.gamma = 0.99
        self.alpha = 0.2
        self.regularization_weight = 0.0
        self.tau = 0.005
        self.max_grad_norm = 1.0
    
    @torch.no_grad()
    def get_action(self, obs:np.ndarray, random:bool=False):
        actor_step:Union[StochasticContinuousPolicyStep, DiscretePolicyStep]  = self.actor(obs)
        
        if random:
            action = actor_step.action.uniform_(-1, 1) * self.action_dim
        else:
            action = actor_step.action
            
        return action
    
    def average_non_zero(self, numbers):
        non_zero_numbers = [num for num in numbers if num != 0]
        if not non_zero_numbers:
            return 0  # Return 0 if there are no non-zero elements
        return sum(non_zero_numbers) / len(non_zero_numbers)
    
    def rollout(self, random:bool=False):
        obs = self.obs
        for i in range(self.rollout_steps):
            action = self.get_action(obs, random)
            next_obs, reward, done, timeout, info = self.envs.step(action)
            
            record = {
                "observations": obs,
                "next_observations": next_obs,
                "actions": action,
                "rewards": reward,
                "dones": done
            }
            
            self.replay_buffer.add_records(record)
            
            obs = next_obs
            
            if "episode" in info:
                print(self.average_non_zero(info['episode']['r']))
        
        self.obs = obs
        
    def update(self, num_iteration:int, batch_size:int):
        for _ in range(num_iteration):
            batch = self.replay_buffer.sample_batch(self.batch_keys, batch_size)
            obs_batch = batch["observations"].to(self.device)
            next_obs_batch = batch["next_observations"].to(self.device)
            action_batch = batch["actions"].to(self.device)
            reward_batch = batch["rewards"].to(self.device)
            done_batch = batch["dones"].to(self.device)
            
            self.critic_optimizer.zero_grad(set_to_none=True)
            print(f"obs_batch: {obs_batch}, action_batch: {action_batch}")
            critic_loss = SAC.compute_critic_loss(self.actor, self.critic, self.critic_target, obs_batch, action_batch, reward_batch,
                                                  next_obs_batch, done_batch, self.alpha, self.gamma)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = False

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss = SAC.compute_policy_loss(self.actor, self.critic, obs_batch.detach(), self.alpha, self.regularization_weight)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = True

            SAC.update_target_param(self.critic, self.critic_target, self.tau)
                
    def train(self, num_epoch:int, num_iteration:int, batch_size:int):
        self.obs, _ = self.envs.reset()
        random = True
        for i in trange(num_epoch):
            if i > (num_epoch // 10):
                random = False
            self.rollout(random)
            self.update(num_iteration, batch_size)
    
    def save_models(self):
        torch.save(self.actor.state_dict(), 'SACactor.pth')
        torch.save(self.critic.state_dict(), 'SACcritic.pth')
        

def main():
    trainer = Trainer()

    trainer.train(num_epoch=500, num_iteration=20, batch_size=512*10)

    trainer.envs.close()
    trainer.save_models()

if __name__ == "__main__":
    main()
    simulation_app.close()