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

"""Rest everything follows."""

import gymnasium
import torch
import torch.optim as optim

from RLAlg.alg.ppo import PPO

from model.modelSAC import Actor

from env.nao_grab_env_cfg import NaoGrabEnvCfg

from RLAlg.nn.steps import StochasticContinuousPolicyStep, DiscretePolicyStep, ValueStep
from typing import Union

import register_env

def process_obs(obs):
    features = obs["policy"]
    return features

class Trainer:
    def __init__(self):
        self.cfg = NaoGrabEnvCfg()
        self.cfg.scene.num_envs = 12

        self.envs = gymnasium.make("NaoGrabEnv-v0", cfg=self.cfg, seed=42)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env_nums, self.obs_dim = self.envs.observation_space.shape

        obs_space = self.envs.observation_space.shape
        action_space = self.envs.action_space.shape

        action_dim = action_space[-1]
        
        self.actor = Actor(self.obs_dim,action_dim, [128, 128], action_dim).to(self.device)
        
        actor_params = torch.load("SACactor.pth")
        self.actor.load_state_dict(actor_params)
        self.actor.eval()

        self.steps = 500
        self.std = 1.0

        self.rollout_steps = self.steps

        self.obs = None

    @torch.no_grad()
    def get_action(self, obs:torch.Tensor):
        actor_step:Union[StochasticContinuousPolicyStep, DiscretePolicyStep]  = self.actor(obs)
        
        action = actor_step.action
        
        return action
    
    def rollout(self):
        obs = self.obs
        for i in range(self.rollout_steps):
            action = self.get_action(obs)
            next_obs, reward, done, timeout, info = self.envs.step(action)
            print(f"Step {i+1}/{self.rollout_steps}, Action: {action}, Reward: {reward}")
            
            
            obs = next_obs
            
            if "episode" in info:
                print(self.average_non_zero(info['episode']['r']))
        
        self.obs = obs
        action = self.get_action(obs)

    def test(self):
        obs, info = self.envs.reset()
        print(obs)
        self.obs = obs
        for epoch in range(1):
            self.rollout()
        
    def check(self):
        obs, info = self.envs.reset()
        print(torch.rad2deg(self.envs.unwrapped.robot.data.joint_pos_limits[0, self.envs.unwrapped.target_joints, 0]))
        print(torch.rad2deg(self.envs.unwrapped.robot.data.joint_pos_limits[0, self.envs.unwrapped.target_joints, 1]))

def main():
    trainer = Trainer()

    trainer.test()
    #trainer.check()

    trainer.envs.close()

if __name__ == "__main__":
    main()
    simulation_app.close()