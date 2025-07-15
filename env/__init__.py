import gymnasium

gymnasium.register(
    id="NAOWalk-v0",
    entry_point=f"{__name__}.walk_env:NAOWalkEnv"
)