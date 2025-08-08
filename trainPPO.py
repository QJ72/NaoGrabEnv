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

from RLAlg.alg.ppo import PPO
from RLAlg.buffer.replay_buffer import ReplayBuffer, compute_gae
from RLAlg.nn.steps import StochasticContinuousPolicyStep, DiscretePolicyStep, ValueStep
from model.modelPPO import Actor, Critic

from env.nao_grab_env_cfg import NaoGrabEnvCfg

import register_env

def process_obs(obs):
    features = obs["policy"]

    return features

class Trainer:
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.envs = gymnasium.make("NaoGrabEnv-v0", cfg=NaoGrabEnvCfg(), seed=42)

        self.env_nums, self.obs_dim = self.envs.observation_space.shape

        self.steps = 50
        self.buffer_steps = 1000
        self.rollout_steps = self.steps

        obs_space = self.envs.observation_space.shape
        action_space = self.envs.action_space.shape

        action_dim = action_space[-1]
        print(f"Observation dim: {self.obs_dim}, Action dim: {action_dim}")


        self.actor = Actor(self.obs_dim,action_dim, [128, 128], max_action=1.0).to(self.device)
        self.critic = Critic(self.obs_dim, [128, 128]).to(self.device)

        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=3e-4
        )
        
        self.replay_buffer = ReplayBuffer(self.env_nums, self.steps, device=self.device)
        self.replay_buffer.create_storage_space("observations", (self.obs_dim,), torch.float32)
        self.replay_buffer.create_storage_space("actions", (action_dim, ), torch.float32)
        self.replay_buffer.create_storage_space("log_probs", (), torch.float32)
        self.replay_buffer.create_storage_space("rewards", (), torch.float32)
        self.replay_buffer.create_storage_space("values", (), torch.float32)
        self.replay_buffer.create_storage_space("dones", (), torch.float32)
        
        self.batch_keys = ["observations", "actions", "log_probs", "rewards", "values", 
                           "dones", "returns", "advantages"]
        
        self.gamma = 0.99
        self.lambda_ = 0.95
        self.clip_ratio = 0.2
        self.regularization_weight = 0.0
        self.max_grad_norm = 0.5
        self.value_loss_weight = 0.5
        self.entropy_weight = 0.01

        self.epoch_rewards = []
        self.step_rewards = []
        self.mex_rewards = []
    
    @torch.no_grad()
    def get_action(self, obs:torch.Tensor):
        actor_step:Union[StochasticContinuousPolicyStep, DiscretePolicyStep]  = self.actor(obs)
        value_step:ValueStep = self.critic(obs)
        
        action = actor_step.action
        log_prob = actor_step.log_prob
        value = value_step.value
        
        return action, log_prob, value
    
    def average_non_zero(self, numbers):
        clone_numbers = numbers.clone().detach()
        return torch.mean(torch.tensor(clone_numbers)[torch.tensor(clone_numbers) != 0]).item() if len(clone_numbers) > 0 else 0.0
    
    
    def rollout(self):
        obs = self.obs
        episode_rewards = []

        for i in range(self.rollout_steps):
            action, log_prob, value = self.get_action(obs)
            next_obs, reward, done, timeout, info = self.envs.step(action)
            
            record = {
                "observations": obs,
                "actions": action,
                "log_probs": log_prob,
                "rewards": reward,
                "values": value,
                "dones": done
            }
            
            self.replay_buffer.add_records(record)
            
            obs = next_obs

            
            #print(F"Episode reward: {self.average_non_zero(record['rewards'])}")
        
        self.obs = obs
        _, _, value = self.get_action(obs)
        returns, advantages = compute_gae(
            self.replay_buffer.data["rewards"],
            self.replay_buffer.data["values"],
            self.replay_buffer.data["dones"],
            value,
            self.gamma,
            self.lambda_
            )
        
        self.replay_buffer.add_storage("returns", returns)
        self.replay_buffer.add_storage("advantages", advantages)

        if "episode" in info:
            avg_episode_reward =self.average_non_zero(info["episode"]["r"])
            episode_rewards.append(info["episode"]["r"])
            #print(f"Average episode reward: {avg_episode_reward:.4f}")
        
        avg_step_reward = self.average_non_zero(self.replay_buffer.data["rewards"])
        self.step_rewards.append(avg_step_reward)


        if episode_rewards:
            self.epoch_rewards.append(self.average_non_zero(episode_rewards))
        else :
            self.epoch_rewards.append(avg_step_reward)
        
        max_reward = self.replay_buffer.data["rewards"].max().item()
        self.mex_rewards.append(max_reward)

    def update(self, num_iteration:int, batch_size:int):
        for _ in range(num_iteration):
            for batch in self.replay_buffer.sample_batchs(self.batch_keys, batch_size):
                obs_batch = batch["observations"].to(self.device)
                action_batch = batch["actions"].to(self.device)
                log_prob_batch = batch["log_probs"].to(self.device)
                value_batch = batch["values"].to(self.device)
                return_batch = batch["returns"].to(self.device)
                advantage_batch = batch["advantages"].to(self.device)

                #print("obs_batch shape:", obs_batch.shape)
                #print("obs_batch min/max:", obs_batch.min().item(), obs_batch.max().item())
                #print("obs_batch contains NaN:", torch.isnan(obs_batch).any().item())

                policy_loss, entropy, kl_divergence = PPO.compute_policy_loss(self.actor, log_prob_batch, obs_batch, action_batch, advantage_batch, self.clip_ratio, self.regularization_weight)

                #print("policy_loss:", policy_loss.item())
                #print("entropy:", entropy.item())
                #print("kl_divergence:", kl_divergence.item())
 

                value_loss = PPO.compute_clipped_value_loss(self.critic, obs_batch, value_batch, return_batch, self.clip_ratio)
                
                loss = policy_loss + value_loss * self.value_loss_weight - entropy * self.entropy_weight

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
    
    def save_models(self):
        torch.save(self.actor.state_dict(), 'PPOactor_without_legs.pth')
        torch.save(self.critic.state_dict(), 'PPOcritic_without_legs.pth')
                
    def train(self, num_epoch:int, num_iteration:int, batch_size:int):
        self.obs, _ = self.envs.reset()
        
        for _ in trange(num_epoch):
            self.rollout()
            self.update(num_iteration, batch_size)

    def plot_rewards(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.epoch_rewards)
        plt.title('Average Rewards Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Average Rewards')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(self.step_rewards)
        plt.title('Step Rewards Over Time')
        plt.xlabel('Rollout')
        plt.ylabel('Step Rewards')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(self.mex_rewards)
        plt.title('Max Rewards per Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Max Reward')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_reward.png', dpi = 300, bbox_inches='tight')
        plt.show()

        print(f"\n Training Summary:")
        print(f"Final Average: {self.epoch_rewards[-1]:.4f}")
        print(f"Best average: {max(self.epoch_rewards):.4f}")
        print(f"Best max reward : {max(self.mex_rewards):.4f}")
        

def main():
    trainer = Trainer()

    trainer.train(num_epoch=100, num_iteration=10, batch_size=512*5)
    trainer.plot_rewards()

    trainer.envs.close()

    trainer.save_models()

if __name__ == "__main__":
    main()
    simulation_app.close()