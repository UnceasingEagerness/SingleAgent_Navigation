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

def run_visitation_heatmap():
    print("--- State Visitation Heatmap (Level 8) ---")
    
    env = SurfaceVesselEnv(show_viewport=False, goal_radius=4.0, obstacle_spacing=43.0, obstacle_scale=(2.0, 5.0), max_steps=1000)

    mock_env = MockEnv()
    actor = Actor(mock_env)
    
    runs_dir = '/home/trizzz/AUV_Project/IISC_Project/SurfaceVessel/SAC-PathPlanning&CollisionAvoidance/SingleAgent_CollisionAvoidance/runs'
    all_runs = glob.glob(os.path.join(runs_dir, '*'))
    
    best_len = 0
    best_run = None
    for r in all_runs:
        csv_path = os.path.join(r, 'reward_components.csv')
        if os.path.exists(csv_path):
            df_len = sum(1 for line in open(csv_path))
            if df_len > best_len:
                best_len = df_len
                best_run = r
    
    checkpoint_path = os.path.join(best_run, "full_checkpoint_best.pth")
    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    clean_actor_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['actor'].items()}
    actor.load_state_dict(clean_actor_state_dict)
    actor.eval()

    episodes = 15
    print(f"Running {episodes} episodes to aggregate visitation density...")
    
    all_x = []
    all_y = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        
        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.get_action(obs_tensor)
            
            action_np = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action_np)
            
            px = info.get("pos_x", None)
            py = info.get("pos_y", None)
            
            if px is not None and py is not None:
                all_x.append(px)
                all_y.append(py)
                
            done = terminated or truncated
            step_count += 1
            
        print(f"Ep {ep+1}/{episodes} finished (Length: {step_count})")

    print(f"Aggregated {len(all_x)} coordinates. Generating Heatmap...")
    
    plt.figure(figsize=(10, 10))
    
    # We use hexbin to create a beautiful density map
    hb = plt.hexbin(all_x, all_y, gridsize=50, cmap='plasma', bins='log', mincnt=1)
    plt.colorbar(hb, label='log10(Visitation Count)')
    
    plt.title('State Visitation Density (Level 8 Analysis)', fontsize=16)
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    
    plt.axis('equal')
    
    output_path = '/home/trizzz/.gemini/antigravity/brain/4c88dfe3-247d-4644-a441-b1035059495f/visitation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved Visitation Heatmap to {output_path}")

if __name__ == "__main__":
    run_visitation_heatmap()
