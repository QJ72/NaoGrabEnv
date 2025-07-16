import numpy as np
import torch

from isaaclab.assets.rigid_object.rigid_object import RigidObject
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms

from .nao_grab_env_cfg import NaoGrabEnvCfg

class NaoGrabEnv(DirectRLEnv):
    cfg:NaoGrabEnvCfg

    def __init__(self, cfg, render_mode = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
    
        self._actions = torch.zeros(self.num_envs, 16, device=self.device)

        self._previous_actions = torch.zeros(self.num_envs, 16, device=self.device)

        self._list_of_joints = [
            "LHipPitch", "RHipPitch", "LShoulderPitch", "LShoulderRoll", "LElbowYaw",
            "LElbowRoll", "LWristYaw","LHand",  "LFinger11",
            "LFinger21", "LThumb1", "LFinger12","LFinger22",
            "LFinger13", "LFinger23", "LThumb2"] #Follow the order in NAO_CFG
        
        self.goal_local = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_world = torch.zeros_like(self.goal_local)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.ball = RigidObject(self.cfg.ball)
        self.box = RigidObject(self.cfg.box)
        self.scene.rigid_objects["ball"] = self.ball
        self.scene.rigid_objects["box"] = self.box

        self.marker = VisualizationMarkers(self.cfg.marker)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing

        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos[self._list_of_joints] #modifs

    def _apply_action(self):
        self.robot.set_joint_position_target(self._processed_actions)

    def _joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[self._list_of_joints] - self.robot.data.default_joint_pos[self._list_of_joints]

    def _distance_hand_to_ball(self, ball_pos) -> torch.Tensor:
        return torch.norm(ball_pos - self._joint_pos()["LHand"])

    def _distance_ball_to_marker(self, ball_pos, marker_pos) -> torch.Tensor:
        return torch.norm(ball_pos - marker_pos)    

    def _get_observations(self):
        self._previous_actions = self._actions.clone()

        joint_pos = self._joint_pos() # (num_envs, 16)
        joint_vel = self.robot.data.joint_vel               # (num_envs, 16)
        previous_actions = self._previous_actions           # (num_envs, 16)
        ball_pos = self.ball.data.root_state_w[:, :3] - self.robot.data.root_state_w[:, :3]

        distance_hand_ball = self._distance_hand_to_ball(ball_pos)

        goal_pos = self.goal_world - self.robot.data.root_pos_w

        distance_ball_marker = self._distance_ball_to_marker(ball_pos, goal_pos)

        obs = torch.cat([
            joint_pos,
            joint_vel,
            previous_actions,
            ball_pos,
            goal_pos,
            distance_hand_ball,
            distance_ball_marker,
        ], dim=-1)


        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return (
            self._no_motion_reward() +
            -0.2 * self._get_action_rate_reward() +
            -0.05 * self._joint_velocity_penalty() +
            - self._distance_hand_to_ball() +
            - self._distance_ball_to_marker()
        )

    def _no_motion_reward(self) -> torch.Tensor:
        lin_vel = torch.norm(self.robot.data.root_lin_vel_b, dim=1)
        ang_vel = torch.norm(self.robot.data.root_ang_vel_b, dim=1)
        penalty = lin_vel**2 + ang_vel**2

        return torch.exp(-penalty / 0.001)
    
    def _get_action_rate_reward(self) -> torch.Tensor:
        return torch.sum((self._actions - self._previous_actions) ** 2, dim=1)
    
    def _joint_velocity_penalty(self) -> torch.Tensor:
        return torch.norm(self.robot.data.joint_vel, dim=1)
    
    def _get_terminate(self):
        ball_height = self.ball.data.root_state_w[:, 2]
        if ball_height < 0.05 :
            return True
        base_height = self.robot.data.root_state_w[:, 2]
        if base_height < 0.15 :
            return True
        return False
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminate = self._get_terminate()
        #terminate = base_height < 0
        return terminate, time_out
    
    def sample_goal_pos(self, env_ids: torch.Tensor):
        x = torch.rand(len(env_ids), device=self.device) * (0.30 - 0.10) + 0.10    # [0.20, 0.10]
        y = torch.rand(len(env_ids), device=self.device) * 0.10
        z = torch.rand(len(env_ids), device=self.device) * 0.2

        goal_local = torch.stack([x, y, z], dim=-1)

        # Update only for the reset envs
        self.goal_local[env_ids] = goal_local

        # Transform to world frame
        root_pos = self.robot.data.root_pos_w[env_ids]
        root_quat = self.robot.data.root_quat_w[env_ids]

        goal_world, _ = combine_frame_transforms(root_pos, root_quat, goal_local)

        self.goal_world[env_ids] = goal_world

        # Visualize (can visualize all or just the updated ones depending on marker implementation)
        self.marker.visualize(self.goal_world)

    
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