import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

project_root = os.path.dirname(os.path.abspath(__file__))

NAO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{project_root}/USDAssets/USDAssets/Nao/Static/nao/nao.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
                # Corrected Joint Names
                "HeadYaw": 0.0, #0
                "HeadPitch": 0.0, #1
                "LHipYawPitch": 0.0, #2
                "LHipRoll": 0.0, #3
                "LHipPitch": 0.0,   #4
                "LKneePitch": 0.0, #5
                "LAnklePitch": 0.0, #6
                "LAnkleRoll": 0.0, #7
                "RHipYawPitch": 0.0, #8
                "RHipRoll": 0.0, #9
                "RHipPitch": 0.0, #10
                "RKneePitch": 0.0,  #11
                "RAnklePitch": 0.0, #12
                "RAnkleRoll": 0.0, #13
                "LShoulderPitch": 0.0,  #14
                "LShoulderRoll": 0.0, #15
                "LElbowYaw": 0.0, #16
                "LElbowRoll": -0.035, #17
                "LWristYaw": 0.0,   #18
                "RShoulderPitch": 0, #19
                "RShoulderRoll": 0.0, #20
                "RElbowYaw": 0.0,   #21
                "RElbowRoll": 0.035, #22
                "RWristYaw": 0.0,  #23
                "LHand": 0.0, #24
                "RHand": 0.0, #25
                "LFinger11": 0.0, #26
                "LFinger21": 0.0, #27
                "RFinger11": 0.0, #28
                "RFinger21": 0.0, #29
                "LThumb1": 0.0, #30
                "RThumb1": 0.0, #31
                "LFinger12": 0.0, #32
                "LFinger22": 0.0, #33
                "RFinger12": 0.0, #34
                "RFinger22": 0.0, #35
                "LFinger13": 0.0, #36
                "LFinger23": 0.0,   #37
                "RFinger13": 0.0, #38
                "RFinger23": 0.0, #39
                "LThumb2": 0.0, #40
                "RThumb2": 0.0, #41
            },
        pos=(0, 0, 0.35)
    ),
    
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                    "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch",
                    "LAnklePitch", "LAnkleRoll", "RHipYawPitch", "RHipRoll",
                    "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll"
                ],
            effort_limit=0,
            velocity_limit=100.0,
            stiffness=100.0,
            damping=10.0
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll",
                    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"
                ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=100.0,
            damping=10.0
        ),

        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "HeadYaw",
                "HeadPitch",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=100.0,
            damping=10.0
        ),

        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                    "LHand", "RHand", "LFinger11", "RFinger11",
                    "LFinger12", "RFinger12", "LFinger21", "RFinger21",
                    "LFinger22", "RFinger22", "LThumb1", "RThumb1",
                    "LThumb2", "RThumb2", "LFinger13", "LFinger23",
                    "RFinger13", "RFinger23", "LWristYaw", "RWristYaw"
                ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=100.0,
            damping=10.0
        )
    }
)