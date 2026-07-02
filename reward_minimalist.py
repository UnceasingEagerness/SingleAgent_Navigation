import numpy as np
from collections import deque

class RewardCalculator:
    def __init__(self, gamma=1.0, k_target=5.0, k_yaw=10.0, k_repulsion=50.0):
        self.gamma = gamma
        self.k_target = k_target
        self.k_yaw = k_yaw
        self.k_repulsion = k_repulsion
        self.prev_potential = None

    def reset(self):
        self.prev_potential = None
        
    def _get_potential(self, dist_to_goal, yaw_error, min_lidar_dist):
        # Potential is negative distance, negative heading error, and negative inverse lidar distance.
        repulsion = (1.0 / max(min_lidar_dist, 0.1)) * self.k_repulsion
        return (-dist_to_goal * self.k_target) - (abs(yaw_error) * self.k_yaw) - repulsion

    def get_minimalist_reward(
        self, pos, goal_pos, lidar_distances_raw, yaw_error,
        goal_radius=4.0, collision_radius_m=3.0
    ):
        """
        A minimalist reward function based on "Reward Misdesign" and "Rewards for Fast Learning".
        No behavior shaping. Only outcomes, steps, and dense subgoals.
        Implemented using strict Potential-Based Reward Shaping (PBRS) to prevent loopholes.
        """
        dist_to_goal = float(np.linalg.norm(goal_pos[:2] - pos[:2]))
        min_lidar_dist = float(np.min(lidar_distances_raw))
        current_potential = self._get_potential(dist_to_goal, yaw_error, min_lidar_dist)
        
        # 1. Base Environment (M) - True Objective
        step_penalty = -1.0
        term_reward = 0.0
        terminated = False
        
        if dist_to_goal < goal_radius:
            term_reward = 1000.0
            terminated = True
        elif np.min(lidar_distances_raw) < collision_radius_m:
            term_reward = -1000.0
            terminated = True

        # 2. Shaping Reward (F) - Ng's PBRS Formula
        shaping_reward = 0.0
        if self.prev_potential is not None and not terminated:
            # F(s, s') = gamma * Phi(s') - Phi(s)
            shaping_reward = (self.gamma * current_potential) - self.prev_potential

        self.prev_potential = current_potential

        # Total Reward
        r_total = term_reward + step_penalty + shaping_reward

        reward_components = {
            "r_target": float(shaping_reward), # Logged as r_target for tensorboard
            "step_penalty": float(step_penalty),
            "r_terminal": float(term_reward)
        }

        return float(r_total), terminated, reward_components
