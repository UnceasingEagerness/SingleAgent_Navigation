import torch
import numpy as np
import time
from surfacevessel_env import SurfaceVesselEnv
from cleanrl_sac import Actor

class MockEnv:
    def __init__(self):
        from gymnasium import spaces
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

def run_inference():
    # 1. Initialize the Environment with rendering enabled
    print("Initializing HoloOcean Environment...")
    # Using a 4.0m goal radius to prevent the physical boat from getting stuck in a turning-radius spiral if it misses the 1.0m mark by a few inches!
    env = SurfaceVesselEnv(show_viewport=True, goal_radius=4.0, obstacle_spacing=50.0, obstacle_scale=(2.0, 5.0), max_steps=5000, spawn_distance_range=(150.0, 200.0))

    # 2. Initialize the Actor Model
    print("Loading Neural Network Weights...")
    mock_env = MockEnv()
    actor = Actor(mock_env)
    
    checkpoint_path = "runs/SurfaceVessel-v0__Phase1_Heading_PBRS_v5__1__1782824169/full_checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    # The saved weights might have '_orig_mod.' prefix if they were compiled using torch.compile. 
    # We strip this prefix to load them safely on CPU without compiling.
    clean_actor_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['actor'].items()}
    actor.load_state_dict(clean_actor_state_dict)
    actor.eval()

    # 3. Run Deterministic Inference Loop
    print("Starting Inference! Watch the Unreal Engine viewport.")
    episodes = 10
    
    all_trajectories = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0
        
        path_x = []
        path_y = []
        goal_x = []
        goal_y = []
        
        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0)
            with torch.no_grad():
                # Inference must be DETERMINISTIC! We extract the mean instead of sampling the distribution.
                mean, _ = actor(obs_tensor)
                y_t = torch.tanh(mean)
                action = y_t * actor.action_scale + actor.action_bias
            
            action_np = action.cpu().numpy()[0]
            action_for_env = action_np
            
            obs, reward, terminated, truncated, info = env.step(action_for_env)
            
            if ep_len % 50 == 0:
                speed = obs[2]
                vel_x = obs[3]
                vel_y = obs[4]
                print(f"Step {ep_len} | Speed: {speed:.2f} m/s | Vel (X,Y): ({vel_x:.2f}, {vel_y:.2f})")

            ep_ret += reward
            ep_len += 1
            
            px = info.get("pos_x", None)
            py = info.get("pos_y", None)
            gx = info.get("goal_x", None)
            gy = info.get("goal_y", None)
            
            if px is not None and py is not None and gx is not None and gy is not None:
                path_x.append(px)
                path_y.append(py)
                goal_x.append(gx)
                goal_y.append(gy)
            
            done = terminated or truncated
            
        print(f"Episode {ep+1} Finished | Return: {ep_ret:.2f} | Length: {ep_len} | Final Distance: {info.get('dist_to_goal', 0):.2f}m")
        
        all_trajectories.append({
            "path_x": np.array(path_x),
            "path_y": np.array(path_y),
            "goal_x": np.array(goal_x),
            "goal_y": np.array(goal_y)
        })
        
    np.save('/home/trizzz/.gemini/antigravity/brain/4c88dfe3-247d-4644-a441-b1035059495f/scratch/inference_trajectories.npy', all_trajectories)
    print("Saved all 10 trajectories to disk!")

    env.close()
    print("Inference completed.")

if __name__ == "__main__":
    run_inference()
