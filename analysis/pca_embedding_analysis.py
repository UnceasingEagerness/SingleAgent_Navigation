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

def run_pca_analysis():
    print("--- Representation Learning Analysis (Level 3) ---")
    env = SurfaceVesselEnv(show_viewport=False, goal_radius=4.0, obstacle_spacing=43.0, obstacle_scale=(2.0, 5.0), max_steps=1000)

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
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(best_run, "actor.pth")
        
    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    if 'actor' in checkpoint:
        clean_actor_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['actor'].items()}
    else:
        clean_actor_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        
    actor.load_state_dict(clean_actor_state_dict)
    actor.eval()

    embeddings = []
    danger_levels = []
    
    episodes = 3
    print("Gathering Lidar Embeddings from 3 episodes...")
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            lidar_input = torch.Tensor(obs[6:]).unsqueeze(0)
            
            with torch.no_grad():
                # Extract the 64-dimensional embedding directly from the lidar perception branch!
                embedding = actor.lidar_net(lidar_input).numpy()[0]
                
            embeddings.append(embedding)
            
            # Record danger level (minimum distance to obstacle, normalized)
            # The lowest value in obs[6:] is the closest obstacle. 0.0 is crash, 1.0 is clear.
            min_dist = np.min(obs[6:])
            danger_levels.append(min_dist)
            
            obs_tensor = torch.Tensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.get_action(obs_tensor)
            
            action_np = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action_np)
            
            done = terminated or truncated

    embeddings = np.array(embeddings)
    danger_levels = np.array(danger_levels)
    
    print("Performing PCA dimensionality reduction using PyTorch SVD...")
    # Center the data
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    emb_mean = emb_tensor.mean(dim=0)
    emb_centered = emb_tensor - emb_mean
    
    # Compute SVD
    U, S, V = torch.svd(emb_centered)
    
    # Project to 2D
    embeddings_2d = torch.mm(emb_centered, V[:, :2]).numpy()
    
    # Calculate explained variance
    variances = (S ** 2) / (emb_centered.shape[0] - 1)
    explained_variance_ratio_ = variances / variances.sum()
    
    plt.figure(figsize=(10, 8))
    # Scatter plot, color coded by how dangerous the state is!
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=danger_levels, cmap='RdYlGn', alpha=0.7, s=30)
    plt.colorbar(scatter, label='Distance to Nearest Obstacle (Red=Crash, Green=Safe)')
    
    plt.title('Lidar Perception Latent Space (PCA Projection)', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({explained_variance_ratio_[0].item()*100:.1f}% Variance)')
    plt.ylabel(f'Principal Component 2 ({explained_variance_ratio_[1].item()*100:.1f}% Variance)')
    plt.grid(True, alpha=0.3)
    
    output_path = '/home/trizzz/.gemini/antigravity/brain/4c88dfe3-247d-4644-a441-b1035059495f/pca_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved PCA Analysis to {output_path}")

if __name__ == "__main__":
    run_pca_analysis()
