from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
import isaaclab.envs.mdp as mdp
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import torch

from .NAO_CFG import NAO_CFG

@configclass
class NaoGrabEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0 #arbitrary

    decimation = 4

    observation_space = 51
    action_space = 17
    state_space = 0

    action_scale = 0.25

    early_termination = True

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.016,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    )

    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0, 0)),
    )

    marker = VisualizationMarkersCfg(
        prim_path="/Visuals/Markers",
        markers={"cylinder": sim_utils.CylinderCfg(
                radius=0.01,
                height=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )}
        )

    scene:InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=12, env_spacing=4.0, replicate_physics=True
    )

    robot:ArticulationCfg = NAO_CFG.replace(prim_path="/World/envs/env_.*/Robot")