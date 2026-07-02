import torch
import numpy as np
import matplotlib.pyplot as plt
from cleanrl_sac import SoftQNetwork
import glob
import os

class MockEnv:
    def __init__(self):
        from gymnasium import spaces
        self.single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(70,))
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

def run_critic_diagnostics():
    print("--- Critic Diagnostics (Level 6) ---")
    mock_env = MockEnv()
    qf1 = SoftQNetwork(mock_env)
    
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
    print(f"Loading Critic weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    if 'qf1' in checkpoint:
        clean_qf1_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['qf1'].items()}
        qf1.load_state_dict(clean_qf1_state_dict)
    
    qf1.eval()

    # We want to create a 2D Heatmap of Q-values.
    # X-axis: Distance to Goal (normalized_dist) [0.0 = At Goal, 1.0 = Far]
    # Y-axis: Distance to Frontal Obstacle [0.0 = Crashing, 1.0 = Clear]
    
    resolution = 50
    dist_to_goal_vals = np.linspace(0.0, 1.0, resolution)
    dist_to_obs_vals = np.linspace(0.0, 1.0, resolution)
    
    Q_values = np.zeros((resolution, resolution))
    
    for i, d_goal in enumerate(dist_to_goal_vals):
        for j, d_obs in enumerate(dist_to_obs_vals):
            # Construct synthetic observation
            # [normalized_dist, sin(yaw), cos(yaw), speed, steering, yaw_rate]
            kinematics = [d_goal, 0.0, 1.0, 0.5, 0.0, 0.0] 
            
            # 64 Lidar beams. 32 is front.
            lidar = np.ones(64)
            # Front obstacle:
            lidar[31] = d_obs
            lidar[32] = d_obs
            lidar[33] = d_obs
            
            obs = np.concatenate([kinematics, lidar])
            action = np.array([0.0, 0.5]) # Go straight, half thrust
            
            obs_tensor = torch.Tensor(obs).unsqueeze(0)
            action_tensor = torch.Tensor(action).unsqueeze(0)
            
            with torch.no_grad():
                q_val = qf1(obs_tensor, action_tensor).item()
                
            Q_values[j, i] = q_val # j is y-axis (obs), i is x-axis (goal)

    plt.figure(figsize=(10, 8))
    # Origin 'lower' puts 0,0 at bottom left
    img = plt.imshow(Q_values, origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap='inferno')
    plt.colorbar(img, label='Predicted Q-Value (Expected Future Reward)')
    
    plt.title('Critic Loss Surface (Value Landscape)', fontsize=16)
    plt.xlabel('Distance to Goal (0=Goal, 1=Far)', fontsize=12)
    plt.ylabel('Distance to Front Obstacle (0=Crash, 1=Clear)', fontsize=12)
    
    output_path = '/home/trizzz/.gemini/antigravity/brain/4c88dfe3-247d-4644-a441-b1035059495f/critic_diagnostics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved Critic Diagnostics Heatmap to {output_path}")

if __name__ == "__main__":
    run_critic_diagnostics()
