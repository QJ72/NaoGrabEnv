import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot (Nao) to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


import os


NAO_USD_PATH = "/home/troja/Desktop/NaoGrabEnv/env/nao_white.usd"


SPHERE_USD_PATH = "/home/troja_robot_lab/Desktop/Lada_Workspace/sphere.usd"


#ValueError: The following joints have default positions out of the limits: 
#	- 'LElbowRoll': 0.000 not in [-1.545, -0.035]
#	- 'RElbowRoll': 0.000 not in [0.035, 1.545]


# Define Nao articulation config
NAO_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{NAO_USD_PATH}"),
    actuators={
        "nao_acts": ImplicitActuatorCfg(
            joint_names_expr=[".*"],          # match all joints
            stiffness=1000.0,                 # increase or tune this value
            damping=50.0,                     # some damping to stabilize
            effort_limit_sim=100.0,           # optional: set joint torque limits
        )
    },
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "LElbowRoll": -0.5,   # within [-1.545, -0.035]
            "RElbowRoll": 0.5,    # within [0.035, 1.545]
        },
        pos=(0.0, 0.0, 0.36),     # ALWAYS SET Z!!!
    )
)


BALL_CONFIG = RigidObjectCfg(
    prim_path="{ENV_NS}/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.16722,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.37447, 0.0, 0.16722),              # ✅ Set position here
        rot=(1.0, 0.0, 0.0, 0.0)          # ✅ Optional: rotation as quaternion (w, x, y, z)
    )
    )



class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    Nao = NAO_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Nao")

    # objects
    Ball = BALL_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Ball")



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    nao = scene["Nao"]
    ball:RigidObject = scene["Ball"]
    joint_names = nao.data.joint_names
    num_joints = len(joint_names)
    idx = joint_names.index("LShoulderPitch")
    idx2 = joint_names.index("RShoulderPitch")

    while simulation_app.is_running():
        scene.reset()

        # Manually set Nao's spawn position (lift it off the ground)
        #root_nao_state = nao.data.default_root_state.clone()
        #root_nao_state[:, 0:3] = torch.tensor([0.0, 0.0, 0.35], device=root_nao_state.device)
        #nao.write_root_state_to_sim(root_nao_state)

        print(ball.data.root_state_w)

        if count > 100:
            # Initialize joint targets to zero or previous values if you want smooth control
            joint_targets = torch.zeros(num_joints, device=scene.device)

            # Compute oscillating target for LShoulderPitch
            target_angle = 0.5 * math.sin(sim_time * 2.0)

 

            # Set target for LShoulderPitch only
            joint_targets[idx] = target_angle
            joint_targets[idx2] = -target_angle  # Set RShoulderPitch to the opposite angle

            # Apply joint position targets for all joints
            nao.set_joint_position_target(joint_targets)

            print(ball.data.root_state_w)
            

        # Write the updated joint data to simulation
        scene.write_data_to_sim()

        # Step the physics
        sim.step()
        sim_time += sim_dt
        count += 1

        # Update scene visuals etc
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

 #   sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    sim.set_camera_view([2.0, 4.0, 1.0], [0.0, 0.0, 0.0])
    # design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
