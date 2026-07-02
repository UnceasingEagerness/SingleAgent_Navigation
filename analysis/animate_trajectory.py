import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from surfacevessel_env import SurfaceVesselEnv
from cleanrl_sac import Actor
import glob
import os

class MockEnv:
    def __init__(self):
        from gymnasium import spaces
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

def animate_trajectory():
    print("--- 2D Trajectory Animation (Success Episode) ---")
    
    env = SurfaceVesselEnv(show_viewport=False, goal_radius=4.0, obstacle_spacing=30.0, obstacle_scale=(2.0, 5.0), max_steps=5000)

    mock_env = MockEnv()
    actor = Actor(mock_env)
    
    # We load the 200k model which we know has successes
    checkpoint_path = "/home/trizzz/AUV_Project/IISC_Project/SurfaceVessel/SAC-PathPlanning&CollisionAvoidance/SingleAgent_CollisionAvoidance/runs/SurfaceVessel-v0__cleanrl_sac__1__1782546981/full_checkpoint_best.pth"
    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    clean_actor_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['actor'].items()}
    actor.load_state_dict(clean_actor_state_dict)
    actor.eval()

    path_x = []
    path_y = []
    goal_x_list = []
    goal_y_list = []
    
    print("Hunting for a successful episode in the chaotic grid (Max 100 tries)...")
    # Keep running episodes until we find a success
    found_success = False
    ep = 0
    while not found_success and ep < 100:
        ep += 1
        obs, info = env.reset()
        done = False
        
        temp_path_x = [0.0]
        temp_path_y = [0.0]
        
        temp_goal_x = []
        temp_goal_y = []
        
        ep_ret = 0
        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.get_action(obs_tensor)
            
            action_np = action.cpu().numpy()[0]
            # Pad 1D action to 2D
            if len(action_np) == 1:
                action_for_env = np.array([0.5, action_np[0]])
            else:
                action_for_env = action_np
                
            obs, reward, terminated, truncated, info = env.step(action_for_env)
            ep_ret += reward
            
            px = info.get("pos_x", None)
            py = info.get("pos_y", None)
            gx = info.get("goal_x", None)
            gy = info.get("goal_y", None)
            
            if px is not None and py is not None and gx is not None and gy is not None:
                temp_path_x.append(px)
                temp_path_y.append(py)
                temp_goal_x.append(gx)
                temp_goal_y.append(gy)
                
            done = terminated or truncated
            
        final_dist = info.get("dist_to_goal", 1000)
        if ep_ret > 0:
            print(f"Found Flawless Success! Episode {ep} | Return: {ep_ret:.2f} | Dist: {final_dist:.2f}m")
            path_x = temp_path_x
            path_y = temp_path_y
            goal_x_list = temp_goal_x
            goal_y_list = temp_goal_y
            found_success = True
            
    if not found_success:
        print("Could not find a successful episode in 100 tries. Using the last one.")
        path_x = temp_path_x
        path_y = temp_path_y
        goal_x_list = temp_goal_x
        goal_y_list = temp_goal_y

    print(f"Animating trajectory of length {len(path_x)} steps...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Agent Successful Evasion & Path Planning', fontsize=14)
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    
    # Plot Start
    ax.scatter(0, 0, color='black', marker='^', s=100, zorder=5, label='Start')
    
    # Plot Exact Physical Obstacles
    exact_obstacles = env.obstacle_coords
    obs_x = [pt[0] for pt in exact_obstacles]
    obs_y = [pt[1] for pt in exact_obstacles]
    ax.scatter(obs_x, obs_y, color='red', alpha=0.2, s=30, label='Obstacle')
            
    ax.set_xlim(min(path_x)-20, max(path_x)+20)
    ax.set_ylim(min(path_y)-20, max(path_y)+20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    line, = ax.plot([], [], color='blue', linewidth=2, label='Path')
    dot, = ax.plot([], [], 'o', color='cyan', markersize=8, markeredgecolor='black', zorder=6, label='AUV')
    goal_dot, = ax.plot([], [], '*', color='gold', markersize=15, markeredgecolor='black', zorder=5, label='Goal')

    ax.legend(loc='upper right')

    # To keep the GIF small, we only animate every Nth frame
    frame_skip = 5
    frames = list(range(0, len(path_x), frame_skip))
    
    def init():
        line.set_data([], [])
        dot.set_data([], [])
        goal_dot.set_data([], [])
        return line, dot, goal_dot

    def update(frame):
        # Frame index could be slightly out of bounds for goal_x_list if arrays mismatched, so we clamp it safely
        g_idx = min(frame, len(goal_x_list)-1)
        line.set_data(path_x[:frame], path_y[:frame])
        dot.set_data([path_x[frame]], [path_y[frame]]) 
        goal_dot.set_data([goal_x_list[g_idx]], [goal_y_list[g_idx]])
        return line, dot, goal_dot

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    
    output_path = '/home/trizzz/.gemini/antigravity/brain/4c88dfe3-247d-4644-a441-b1035059495f/trajectory_animation.gif'
    
    print("Saving GIF using PillowWriter...")
    # Try using pillow which is built into python/matplotlib without needing external ffmpeg
    writer = animation.PillowWriter(fps=20)
    ani.save(output_path, writer=writer)
    
    print(f"Saved Animation to {output_path}")

if __name__ == "__main__":
    animate_trajectory()
