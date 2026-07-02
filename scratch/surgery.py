import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append(".")
from cleanrl_sac import Actor, SoftQNetwork
from surfacevessel_env import SurfaceVesselEnv

def main():
    print("Loading 1D models...")
    # These are the weights from the Phase 2 Marathon
    checkpoint = torch.load("runs/SurfaceVessel-v0__cleanrl_sac__1__1782546981/full_checkpoint_final.pth")
    old_actor_dict = checkpoint["actor"]
    old_qf1_dict = checkpoint["qf1"]
    old_qf2_dict = checkpoint["qf2"]

    print("Instantiating 2D networks...")
    env = SurfaceVesselEnv() # Automatically has action_dim=2 now
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space
    new_actor = Actor(env)
    new_qf1 = SoftQNetwork(env)
    new_qf2 = SoftQNetwork(env)

    def modify_actor(old_dict, new_model):
        new_dict = new_model.state_dict()
        for name, param in old_dict.items():
            name = name.replace("_orig_mod.", "")
            if "fc_mean.weight" in name or "fc_logstd.weight" in name:
                # new_dict is [2, 128]. old is [1, 128]
                new_dict[name][1] = param[0] # Map old steering to new steering
                # throttle (index 0) remains random
            elif "fc_mean.bias" in name or "fc_logstd.bias" in name:
                new_dict[name][1] = param[0]
                if "fc_mean" in name:
                    new_dict[name][0] = 0.8 # Initialize throttle bias to encourage moving forward initially
            elif "action_scale" in name or "action_bias" in name:
                new_dict[name] = new_model.state_dict()[name]
            else:
                new_dict[name] = param
        return new_dict

    def modify_qf(old_dict, new_model):
        new_dict = new_model.state_dict()
        for name, param in old_dict.items():
            name = name.replace("_orig_mod.", "")
            if "q_net.0.weight" in name:
                # new_dict is [256, 130]. old is [256, 129]
                new_dict[name][:, :128] = param[:, :128] # Fusion features
                new_dict[name][:, 129] = param[:, 128]   # Map old steering to new steering
                # throttle (index 128) remains random
            else:
                new_dict[name] = param
        return new_dict

    print("Performing surgery...")
    new_actor.load_state_dict(modify_actor(old_actor_dict, new_actor))
    new_qf1.load_state_dict(modify_qf(old_qf1_dict, new_qf1))
    new_qf2.load_state_dict(modify_qf(old_qf2_dict, new_qf2))

    print("Saving 2D weights...")
    torch.save(new_actor.state_dict(), "actor_2d_init.pth")
    torch.save(new_qf1.state_dict(), "qf1_2d_init.pth")
    torch.save(new_qf2.state_dict(), "qf2_2d_init.pth")
    print("Surgery Complete! Successfully generated 2D Action Space weights.")

if __name__ == "__main__":
    main()
