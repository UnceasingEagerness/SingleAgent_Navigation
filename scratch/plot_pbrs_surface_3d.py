import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_full_pbrs_landscape():
    # Grid: Distance to Goal (X) vs Min LiDAR Dist (Y)
    dist_to_goal = np.linspace(0, 50, 100)
    min_lidar = np.linspace(0, 15, 100)
    X, Y = np.meshgrid(dist_to_goal, min_lidar)
    
    # Assume the agent moves 1.0m towards the goal per step on average
    prev_dist = X + 1.0 
    
    Z = np.zeros_like(X)
    
    gamma = 0.99
    k_target = 5.0
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            c_dist = X[i, j]
            c_lidar = Y[i, j]
            p_dist = prev_dist[i, j]
            
            # Terminal Conditions
            if c_dist < 4.0:
                Z[i, j] = 1000.0  # Goal Reached!
            elif c_lidar < 3.0:
                Z[i, j] = -1000.0 # Crash!
            else:
                # Normal PBRS Step
                Phi_curr = -c_dist * k_target
                Phi_prev = -p_dist * k_target
                shaping = (gamma * Phi_curr) - Phi_prev
                Z[i, j] = shaping - 1.0 # shaping + step_penalty
                
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none', alpha=0.9)
    
    ax.set_xlabel('Distance to Goal (m)')
    ax.set_ylabel('Min LiDAR Distance (m)')
    ax.set_zlabel('Total Step Reward')
    ax.set_title('Full PBRS Reward Landscape (Including Terminals)')
    
    # Adjust viewing angle for maximum "3D-ness"
    ax.view_init(elev=30, azim=-45)
    
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Reward')
    
    plt.tight_layout()
    plt.savefig('/home/trizzz/.gemini/antigravity/brain/4c88dfe3-247d-4644-a441-b1035059495f/new_reward_surface_3d.png', dpi=300)
    print("Saved to new_reward_surface_3d.png")

if __name__ == '__main__':
    plot_full_pbrs_landscape()
