import torch
import numpy as np
import matplotlib.pyplot as plt
from surfacevessel_env import SurfaceVesselEnv
from cleanrl_sac import Actor
import glob
import os

class MockEnv:
    def __init__(self):
        from gymnasium import spaces
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

def plot_trajectories():
    print("--- 2D Trajectory Analysis ---")
    print("Initializing environment...")
    
    # We use obstacle_spacing=43.0 and goal_radius=4.0
    env = SurfaceVesselEnv(show_viewport=False, goal_radius=4.0, obstacle_spacing=43.0, obstacle_scale=(2.0, 5.0), max_steps=1000)

    print("Loading Best Neural Network Weights (from 300k run)...")
    mock_env = MockEnv()
    actor = Actor(mock_env)
    
    runs_dir = '/home/trizzz/AUV_Project/IISC_Project/SurfaceVessel/SAC-PathPlanning&CollisionAvoidance/SingleAgent_CollisionAvoidance/runs'
    all_runs = glob.glob(os.path.join(runs_dir, '*'))
    
    # Let's find the run with the largest CSV (the 300k run)
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
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(best_run, "actor.pth")
        
    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    # Handle whether it's a full checkpoint or just actor weights
    if 'actor' in checkpoint:
        clean_actor_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['actor'].items()}
    else:
        clean_actor_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        
    actor.load_state_dict(clean_actor_state_dict)
    actor.eval()

    print("Running Evaluation Episodes...")
    episodes = 5
    
    plt.figure(figsize=(10, 10))
    colors = ['blue', 'green', 'purple', 'orange', 'cyan']
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        
        path_x = []
        path_y = []
        
        goal_x = info.get("goal_x", 0.0)
        goal_y = info.get("goal_y", 0.0)
        
        # We start at 0,0
        path_x.append(0.0)
        path_y.append(0.0)
        
        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.get_action(obs_tensor)
            
            action_np = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action_np)
            
            # The environment was updated to return pos_x and pos_y
            px = info.get("pos_x", None)
            py = info.get("pos_y", None)
            
            if px is not None and py is not None:
                path_x.append(px)
                path_y.append(py)
            
            done = terminated or truncated
            
        print(f"Episode {ep+1} Path Length: {len(path_x)} steps")
        
        # Plot this trajectory
        plt.plot(path_x, path_y, color=colors[ep], alpha=0.7, linewidth=2, label=f'Ep {ep+1} Trajectory')
        
        # Plot Goal
        plt.scatter(goal_x, goal_y, color='gold', marker='*', s=300, edgecolor='black', zorder=5)
        
        # Plot Start
        plt.scatter(0, 0, color='black', marker='^', s=100, zorder=5)

    # We cannot perfectly plot the procedural obstacles because they are generated inside the UE plugin via C++
    # But we CAN plot a grid representing the general 43.0m obstacle spacing to give a sense of the maze
    grid_size = 300
    spacing = 43.0
    for x in np.arange(-grid_size, grid_size, spacing):
        for y in np.arange(-grid_size, grid_size, spacing):
            plt.scatter(x, y, color='red', alpha=0.1, s=20)
            
    plt.title('Agent 2D Behavioral Trajectories (Level 9 Analysis)', fontsize=16)
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.grid(True, alpha=0.3)
    
    # Fix axes to be equal so the curvature isn't distorted
    plt.axis('equal')
    
    output_path = '/home/trizzz/.gemini/antigravity/brain/4c88dfe3-247d-4644-a441-b1035059495f/trajectories.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved Trajectories graph to {output_path}")

if __name__ == "__main__":
    plot_trajectories()
