import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_pbrs_surface():
    # Define parameters
    gamma = 0.99
    k_target = 5.0
    
    # Create a grid of previous and current distances (from 0 to 100 meters)
    prev_dists = np.linspace(0, 100, 100)
    curr_dists = np.linspace(0, 100, 100)
    
    P, C = np.meshgrid(prev_dists, curr_dists)
    
    # Calculate Potential: Phi(s) = -dist * k_target
    Phi_prev = -P * k_target
    Phi_curr = -C * k_target
    
    # Calculate PBRS Shaping Reward: F = gamma * Phi(s') - Phi(s)
    # Plus the base step penalty (-1.0)
    R_total = (gamma * Phi_curr) - Phi_prev - 1.0
    
    # Plot the surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(P, C, R_total, cmap='viridis', edgecolor='none', alpha=0.9)
    
    # Also plot the old clipped reward for comparison (ghosted out)
    # progress = P - C
    # old_R_target = np.clip(progress, -0.5, 10.0) * k_target
    # old_R_total = old_R_target - 1.0
    # ax.plot_surface(P, C, old_R_total, cmap='Reds', edgecolor='none', alpha=0.3)
    
    ax.set_xlabel('Previous Distance to Goal (m)')
    ax.set_ylabel('Current Distance to Goal (m)')
    ax.set_zlabel('Total Step Reward')
    ax.set_title('New PBRS Reward Surface (gamma=0.99, k=5.0)')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.savefig('/home/trizzz/.gemini/antigravity/brain/4c88dfe3-247d-4644-a441-b1035059495f/new_reward_surface.png')
    print("Saved to new_reward_surface.png")

if __name__ == '__main__':
    plot_pbrs_surface()
