import torch
import numpy as np
import matplotlib.pyplot as plt
from cleanrl_sac import Actor
import glob
import os

class MockEnv:
    def __init__(self):
        from gymnasium import spaces
        self.single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(70,))
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

def run_saliency():
    print("--- Jacobian Saliency (Level 12) ---")
    mock_env = MockEnv()
    actor = Actor(mock_env)
    
    runs_dir = '/home/trizzz/AUV_Project/IISC_Project/SurfaceVessel/SAC-PathPlanning&CollisionAvoidance/SingleAgent_CollisionAvoidance/runs'
    all_runs = glob.glob(os.path.join(runs_dir, '*'))
    
    best_csv = None
    max_len = 0
    best_run = None
    for r in all_runs:
        csv_path = os.path.join(r, 'reward_components.csv')
        if os.path.exists(csv_path):
            df_len = sum(1 for line in open(csv_path))
            if df_len > max_len:
                max_len = df_len
                best_run = r
    
    checkpoint_path = os.path.join(best_run, "full_checkpoint_best.pth")
    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    clean_actor_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['actor'].items()}
    actor.load_state_dict(clean_actor_state_dict)
    actor.eval()

    # We want to compute ∂action/∂lidar
    # Let's create a scenario: An obstacle is placed very close on the front-left (beam 28)
    kinematics = np.array([0.5, 0.0, 1.0, 0.5, 0.0, 0.0], dtype=np.float32)
    lidar = np.ones(64, dtype=np.float32)
    
    # Obstacle on the left (beams 15-25 are left, 32 is front, 45-55 are right)
    # The Lidar sweeps 360 degrees. 64 beams = 5.625 degrees per beam.
    # Front is around beam 32. Left is around beam 16. Right is around beam 48.
    
    obs = np.concatenate([kinematics, lidar])
    obs_tensor = torch.Tensor(obs).unsqueeze(0)
    obs_tensor.requires_grad_(True)
    
    # Forward pass
    action, _ = actor.get_action(obs_tensor)
    
    # action[0, 0] is throttle, action[0, 1] is steering
    steering = action[0, 1]
    
    # Compute gradients of steering with respect to observation
    steering.backward()
    
    gradients = obs_tensor.grad.numpy()[0]
    
    # Extract just the lidar gradients
    lidar_grads = gradients[6:]
    
    plt.figure(figsize=(12, 6))
    
    # Beams 0 to 63
    beams = np.arange(64)
    # The gradient tells us: If this lidar beam value INCREASES, how does steering change?
    # If gradient is POSITIVE, increasing distance (clearer path) turns the boat RIGHT (+1)
    # Or in other words, a rock appearing (distance DECREASES) turns the boat LEFT (-1)
    
    plt.bar(beams, np.abs(lidar_grads), color='blue', alpha=0.7)
    
    # Highlight front, left, right
    plt.axvspan(28, 36, color='red', alpha=0.2, label='Front Beams')
    plt.axvspan(12, 20, color='green', alpha=0.2, label='Left Beams')
    plt.axvspan(44, 52, color='orange', alpha=0.2, label='Right Beams')
    
    plt.title('Jacobian Saliency: | ∂(Steering) / ∂(Lidar Beam) |', fontsize=16)
    plt.xlabel('Lidar Beam Index (0 to 63)', fontsize=12)
    plt.ylabel('Absolute Sensitivity', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = '/home/trizzz/.gemini/antigravity/brain/4c88dfe3-247d-4644-a441-b1035059495f/jacobian_saliency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved Saliency Map to {output_path}")

if __name__ == "__main__":
    run_saliency()
