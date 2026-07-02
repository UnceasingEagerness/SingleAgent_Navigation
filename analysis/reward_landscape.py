import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_reward_landscape():
    print("--- Plotting Mathematical Reward Landscape ---")
    
    # 1. Define the input ranges
    # Progress: How much closer we get to the goal in one step (meters)
    # A fast boat might cover 0.5m in a single frame.
    progress_vals = np.linspace(-0.5, 0.5, 100) 
    
    # Nearest Obstacle: Distance to the closest rock (meters)
    obstacle_dist_vals = np.linspace(0.0, 10.0, 100)
    
    # Create meshgrid for 3D plotting
    P, O = np.meshgrid(progress_vals, obstacle_dist_vals)
    
    # 2. Compute the Reward Function
    # R = (progress * k_target) + step_penalty
    # If O < 3.0: R = -1000.0
    k_target = 5.0
    step_penalty = -1.0
    collision_radius = 3.0
    
    # Calculate base reward
    R = (np.clip(P, -0.5, 10.0) * k_target) + step_penalty
    
    # Apply collision cliff
    R[O < collision_radius] = -1000.0
    
    # 3. Create 3D Surface Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # We clip the bottom of the graph to -20 instead of -1000 so the cliff is visible but doesn't dwarf the progress slope
    R_clipped = np.clip(R, -20.0, 5.0)
    
    surf = ax.plot_surface(P, O, R_clipped, cmap='coolwarm', edgecolor='none', alpha=0.8)
    
    ax.set_title('Minimalist Reward Landscape (3D Surface)', fontsize=16)
    ax.set_xlabel('Progress Towards Goal (meters/step)', fontsize=12)
    ax.set_ylabel('Nearest Obstacle Distance (meters)', fontsize=12)
    ax.set_zlabel('Immediate Reward (r_t)', fontsize=12)
    
    # Highlight the cliff
    ax.view_init(elev=20, azim=135)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Reward')
    
    output_path = '/home/trizzz/.gemini/antigravity/brain/4c88dfe3-247d-4644-a441-b1035059495f/reward_landscape.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D Reward Landscape to {output_path}")

if __name__ == "__main__":
    plot_reward_landscape()
