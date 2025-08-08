import numpy as np
import torch
import random

import time 

import os

from isaaclab.assets.rigid_object.rigid_object import RigidObject
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.sensors import FrameTransformer

from .nao_grab_env_cfg import NaoGrabEnvCfg

class NaoGrabEnv(DirectRLEnv):
    cfg:NaoGrabEnvCfg

    def __init__(self, cfg, render_mode = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.target_joints = torch.tensor(
            [2,7,11,15,19,25, 23,24,26,32,33,38,39,34],
            dtype=torch.long,
            device=self.device
        )
    
        self._actions = torch.zeros(self.num_envs, len(self.target_joints), device=self.device)

        self._previous_actions = torch.zeros_like(self._actions)
        
        self.goal_local = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_world = torch.zeros_like(self.goal_local)


        self.ball_local = torch.zeros(self.num_envs, 3, device=self.device)
        self.ball_world = torch.zeros_like(self.goal_local)

        self._processed_actions = torch.zeros(self.num_envs, 
                                                   self.robot.data.joint_pos.shape[1], 
                                                   device=self.device)

    def _local_to_global_position(self, local_pos: torch.Tensor, envs_id: torch.Tensor) -> torch.Tensor:
        """Convert local position to global position based on the robot's root position."""
        #print(f"envs_id: {envs_id}, type: {type(envs_id)}")
        #print(f"local_pos: {local_pos}, type: {type(local_pos)}")
        idx = None
        if envs_id.__class__ == int:
            idx = envs_id
            #print(f"Single envs_id: {idx}")
        elif isinstance(envs_id, torch.Tensor) and envs_id.numel() == 1:
            idx = int(envs_id.cpu().item())
            #print(f"Single-element envs_id: {idx}")
        if idx is not None:
            #print(f"Using idx: {idx}")
            #print(f"Root position: {self.robot.data.root_pos_w[idx]}")

            final_local_pos = local_pos.clone().to(self.device) + torch.tensor([
                    self.robot.data.root_pos_w[idx][0],
                    self.robot.data.root_pos_w[idx][1],
                    0.0
                ], device=self.device)
            return final_local_pos
        local_pos = local_pos.clone().to(self.device)
        for i in range(len(envs_id)):
            local_pos[i, 0] += self.robot.data.root_pos_w[envs_id[i], 0]
            local_pos[i, 1] += self.robot.data.root_pos_w[envs_id[i], 1]
        return local_pos
    
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.ball = RigidObject(self.cfg.ball)
        self.box = RigidObject(self.cfg.box)
        self.scene.rigid_objects["ball"] = self.ball
        self.scene.rigid_objects["box"] = self.box

        self.end_frame = FrameTransformer(self.cfg.end_effector)
        self.scene.sensors["frame"] = self.end_frame
        self.end_effector_marker = VisualizationMarkers(self.cfg.marker)

        self.marker = VisualizationMarkers(self.cfg.marker)
        
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing

        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.previous_joints_pos = self.robot.data.joint_pos.clone()
        self._actions = actions.clone()
        #print(f"Actions: {self._actions}")
        full_actions = torch.zeros_like(self.robot.data.default_joint_pos, device=self.device)
        full_actions[:, self.target_joints] = self._actions
        self._processed_actions = self.cfg.action_scale * full_actions + self.robot.data.joint_pos
        #print(f"joints position: {self.robot.data.joint_pos[:, self._indice_joints]}")
        #print(f"Processed actions: {self._processed_actions}")
        #print(self._processed_actions)

    def _apply_action(self):

        self.robot.set_joint_position_target(self._processed_actions) 

    def _get_observations(self):
        self._previous_actions = self._actions.clone()

        joint_pos = self.robot.data.joint_pos- self.robot.data.default_joint_pos


        joint_pos = joint_pos[:, self.target_joints] # (num_envs, 16)

        joint_vel = self.robot.data.joint_vel[:, self.target_joints]            # (num_envs, 16)
        ball_pos = self.ball_world # (num_envs, 3)

        ee_pos = self.end_frame.data.target_pos_w.squeeze(1)

        #print(f"goal world: {self.goal_world}, Robot root position: {self.robot.data.root_pos_w}")
        #goal_pos = self.goal_world - self.robot.data.root_pos_w

        obs = torch.cat([
            joint_pos,
            joint_vel,
            self._previous_actions,
            ball_pos,
            ee_pos,
            #goal_pos,
        ], dim=-1)


        return obs

    def _get_rewards(self) -> torch.Tensor:
        ee_pos = self.end_frame.data.target_pos_w.squeeze(1)

        #print(f"End effector position: {ee_pos}")
        #print(f"Ball world position: {self.ball_world}")
        self.marker.visualize(self.ball_world)

        distance_hand_ball = torch.norm(ee_pos - self.ball_world, dim=1)
        #print(f"distance: {distance_hand_ball}")

        scale = 0.2
        reward_tracking_pos = 5.0 * distance_hand_ball

        time.sleep(0.5)

        success_bonus = (distance_hand_ball < 0.1).float() * 2.0

        #self.distance_ball_marker = self._distance(self.ball_world, self.goal_world)

        #reward = self._no_motion_reward() + (-0.01 * self._get_action_rate_reward()) + (-0.001 * self._joint_velocity_penalty()) + 5*reward_tracking_pos
        reward = reward_tracking_pos + success_bonus

        return reward

    def _no_motion_reward(self) -> torch.Tensor:
        lin_vel = torch.norm(self.robot.data.root_lin_vel_b, dim=1)
        ang_vel = torch.norm(self.robot.data.root_ang_vel_b, dim=1)
        penalty = lin_vel**2 + ang_vel**2

        return torch.exp(-penalty / 0.001)
    
    def _get_action_rate_reward(self) -> torch.Tensor:
        return torch.sum((self._actions - self._previous_actions) ** 2, dim=1)
    
    def _joint_velocity_penalty(self) -> torch.Tensor:
        return torch.norm(self.robot.data.joint_vel, dim=1)
    
    def _get_terminate(self) -> bool:
        ball_height = self.ball.data.root_state_w[:, 2]
        base_height = self.robot.data.root_state_w[:, 2]

        terminate = (ball_height < 0.05) | (base_height < 0.2)

        return terminate
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminate = self._get_terminate()
        return terminate, time_out
    
    def sample_ball_pos(self, env_ids: torch.Tensor):

        x = torch.rand(len(env_ids), device=self.device)*0.1 + 0.2 #torch.rand(len(env_ids), device=self.device) * 0.15 + 0.15
        y = torch.rand(len(env_ids), device=self.device)*0.15 +0.1 #torch.rand(len(env_ids), device=self.device)* (-0.1)  # [-0.10, 0]
        z = torch.ones(len(env_ids), device=self.device)*0.22 # Constant height


        ball_local = torch.stack([x, y, z], dim=-1)

        #print(f"Ball local: {ball_local}")
        # Update only for the reset envs
        self.ball_local[env_ids] = ball_local

        ball_world = self._local_to_global_position(ball_local, env_ids)

        self.ball_world[env_ids] = ball_world
        
        default_quat = self.ball.data.default_root_state[env_ids, 3:7]

        ball_pose = torch.cat([ball_world, default_quat], dim=1)

        self.ball.write_root_pose_to_sim(ball_pose, env_ids=env_ids)

        default_velocity = torch.zeros(len(env_ids), 6, device=self.device)
        self.ball.write_root_velocity_to_sim(default_velocity, env_ids=env_ids)

    def reset_box_pos(self, env_ids: torch.Tensor | None = None):
        box_start_position = self.box.data.default_root_state[env_ids, :7]  # Only position + quaternion
        #print(f"Box start position: {box_start_position}")
        #print(f"env_ids: {env_ids}, type: {type(env_ids)}")
        #print(f"Terrain origins: {self.terrain.env_origins}")
        if env_ids.numel() == 1:
            #print(f"box start position before adjustment: {box_start_position[0, 0]}, {box_start_position[0, 1]}")
            #print(f"Terrain origin for env {env_ids[0]}: {self.terrain.env_origins[env_ids, 0]}")
            box_start_position[0, 0] += self.terrain.env_origins[env_ids[0], 0]
            box_start_position[0, 1] += self.terrain.env_origins[env_ids[0], 1]
        else:
            # For multiple environments, adjust the box position for each environment
            for i in range(len(env_ids)):
                box_start_position[i, 0] += self.terrain.env_origins[env_ids[i]][0]
                box_start_position[i, 1] += self.terrain.env_origins[env_ids[i]][1]

        self.box.write_root_pose_to_sim(box_start_position, env_ids=env_ids)


    def _generate_goal_local(self, env_ids: torch.Tensor) -> torch.Tensor:
        x = torch.tensor([0.2 for _ in range(len(env_ids))], device=self.device) #torch.rand(len(env_ids), device=self.device) * 0.2 + 0.2   # [0.30, 0.15]
        y = torch.tensor([0.15 for _ in range(len(env_ids))], device=self.device) #torch.rand(len(env_ids), device=self.device) * 0    # [0.20, 0.10]
        z = torch.tensor([0.216 for _ in range(len(env_ids))], device=self.device)  # Constant height

        return torch.stack([x, y, z], dim=-1)

    def sample_goal_pos(self, env_ids: torch.Tensor):
        goal_local = self._generate_goal_local(env_ids)

        #print(f"Goal local: {goal_local}")

        # Update only for the reset envs
        self.goal_local[env_ids] = goal_local

        self.goal_world[env_ids] = self._local_to_global_position(goal_local, env_ids)

        # Visualize (can visualize all or just the updated ones depending on marker implementation)
        self.marker.visualize(self.goal_world[env_ids])
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
       
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.previous_joints_pos = self.robot.data.joint_pos.clone()
        self.sample_goal_pos(env_ids)
        self.reset_box_pos(env_ids)
        self.sample_ball_pos(env_ids)