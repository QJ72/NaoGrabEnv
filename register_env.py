from env.nao_grab_env import NaoGrabEnv
import gymnasium

gymnasium.register(
    id='NaoGrabEnv-v0',
    entry_point='env.nao_grab_env:NaoGrabEnv',
)