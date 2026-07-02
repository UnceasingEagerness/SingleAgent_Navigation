import torch
import numpy as np
from surfacevessel_env import SurfaceVesselEnv
from cleanrl_sac import Actor
import glob
import os

class MockEnv:
    def __init__(self):
        from gymnasium import spaces
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

def run_generalization():
    print("--- Generalization Test (Level 10) ---")
    
    # NORMAL spacing is 43.0. DENSE spacing is 25.0! (almost double the obstacles)
    env = SurfaceVesselEnv(show_viewport=False, goal_radius=4.0, obstacle_spacing=25.0, obstacle_scale=(2.0, 5.0), max_steps=1000)

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

    episodes = 30
    successes = 0
    crashes = 0
    timeouts = 0
    
    print(f"\nRunning {episodes} episodes in DENSE obstacle mode (spacing=25.0)...")
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.get_action(obs_tensor)
            
            action_np = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            
        # Old reward logic check
        if reward > 500:
            successes += 1
            print(f"Ep {ep+1}: SUCCESS")
        elif reward < -500:
            crashes += 1
            print(f"Ep {ep+1}: CRASH")
        else:
            timeouts += 1
            print(f"Ep {ep+1}: TIMEOUT")

    print("\n--- Out-of-Distribution (OOD) Results ---")
    print(f"Success Rate: {(successes/episodes)*100:.1f}%")
    print(f"Crash Rate: {(crashes/episodes)*100:.1f}%")
    print(f"Timeout Rate: {(timeouts/episodes)*100:.1f}%")
    
    # Save a text report
    with open('/home/trizzz/.gemini/antigravity/brain/4c88dfe3-247d-4644-a441-b1035059495f/generalization_results.txt', 'w') as f:
        f.write(f"OOD Generalization (Obstacle Spacing 25.0 instead of 43.0)\n")
        f.write(f"Success Rate: {(successes/episodes)*100:.1f}%\n")
        f.write(f"Crash Rate: {(crashes/episodes)*100:.1f}%\n")
        f.write(f"Timeout Rate: {(timeouts/episodes)*100:.1f}%\n")

if __name__ == "__main__":
    run_generalization()
