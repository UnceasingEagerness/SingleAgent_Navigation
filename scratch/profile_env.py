import cProfile
import pstats
import gymnasium as gym
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surfacevessel_env import SurfaceVesselEnv

def run_sim():
    env = SurfaceVesselEnv(max_steps=1000, show_viewport=False)
    obs, _ = env.reset()
    for _ in range(2000): # run 2000 steps to get a good profile
        action = env.action_space.sample()
        obs, rewards, term, trunc, infos = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
    env.close()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run_sim()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(40)
