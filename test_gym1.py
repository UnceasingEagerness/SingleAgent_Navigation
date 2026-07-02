import gymnasium as gym
from surfacevessel_env import SurfaceVesselEnv
import numpy as np

def make_env(seed, idx):
    def thunk():
        env = SurfaceVesselEnv(max_steps=5, show_viewport=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

envs = gym.vector.AsyncVectorEnv([make_env(0, 0), make_env(0, 1)])
obs, _ = envs.reset()
for i in range(10):
    actions = np.array([[0.0], [0.0]])
    obs, rewards, term, trunc, infos = envs.step(actions)
    if any(term) or any(trunc):
        print("Keys in infos:", infos.keys())
        if "episode" in infos:
            print("Episode info:", infos["episode"])
        if "final_info" in infos:
            print("Final info:", infos["final_info"])
        if "_episode" in infos:
            print("_episode flag:", infos["_episode"])
        break
envs.close()
