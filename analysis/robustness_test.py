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

def run_robustness():
    print("--- Robustness Test (Level 11) ---")
    
    # NORMAL spacing is 43.0
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

    episodes = 30
    successes = 0
    crashes = 0
    timeouts = 0
    
    print(f"\nRunning {episodes} episodes with 20% Action Noise...")
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.get_action(obs_tensor)
            
            action_np = action.cpu().numpy()[0]
            
            # --- INJECT NOISE ---
            # action[0] is throttle, action[1] is steering
            # We inject N(0, 0.2) noise into steering to simulate ocean currents
            noise = np.random.normal(0, 0.2)
            action_np[1] = np.clip(action_np[1] + noise, -1.0, 1.0)
            
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            
        if reward > 500:
            successes += 1
            print(f"Ep {ep+1}: SUCCESS")
        elif reward < -500:
            crashes += 1
            print(f"Ep {ep+1}: CRASH")
        else:
            timeouts += 1
            print(f"Ep {ep+1}: TIMEOUT")

    print("\n--- Robustness (Noise Injection) Results ---")
    print(f"Success Rate: {(successes/episodes)*100:.1f}%")
    print(f"Crash Rate: {(crashes/episodes)*100:.1f}%")
    print(f"Timeout Rate: {(timeouts/episodes)*100:.1f}%")
    
    with open('/home/trizzz/.gemini/antigravity/brain/4c88dfe3-247d-4644-a441-b1035059495f/robustness_results.txt', 'w') as f:
        f.write(f"Robustness Test (20% Gaussian Action Noise on Steering)\n")
        f.write(f"Success Rate: {(successes/episodes)*100:.1f}%\n")
        f.write(f"Crash Rate: {(crashes/episodes)*100:.1f}%\n")
        f.write(f"Timeout Rate: {(timeouts/episodes)*100:.1f}%\n")

if __name__ == "__main__":
    run_robustness()
