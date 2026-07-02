#The most feasible thing to do to get reward is to go near the goal to minimise the distance penalty and keep circling in order to get the velocity bonus - Adjusting the scaling removed it


import numpy as np
from collections import deque

class SurfaceVesselReward:
    def __init__(self):
        self.prev_dist = None
        self.prev_delta = 0.0
        self.action_history = deque(maxlen=20)
        self.prev_steering = 0.0
        self._beam_weights = None

    def reset(self):
        """Reset the internal state at the start of each episode."""
        self.prev_dist = None
        self.prev_delta = 0.0
        self.action_history.clear()
        self.prev_steering = 0.0

    def get_reward(self, pos, vel, rpy, goal_pos, action_diff,goal_radius):
        """Calculates the reward based on the current state."""
        dist = float(np.linalg.norm(goal_pos[:2] - pos[:2]))
        speed = float(np.linalg.norm(vel[:2]))

        reward = 0.0

        # 1. Linear distance penalty
        reward -= 100 * dist    #Increasing this from 0.1 to 100 made it just go near the goal and cirle and never reach it

        # 2. Progress reward
        if self.prev_dist is not None:
            progress = self.prev_dist - dist
            safe_progress = np.clip(progress, -2.0, 2.0) 
            reward += 50.0 * safe_progress
        self.prev_dist = dist

        # 3. Heading alignment 
        goal_dir_yaw = np.arctan2(
            goal_pos[1] - pos[1],
            goal_pos[0] - pos[0]
        )
        
        vessel_yaw_rad = np.deg2rad(rpy[2]) 
        yaw_error = abs((goal_dir_yaw - vessel_yaw_rad + np.pi) % (2 * np.pi) - np.pi)
        
        forward_alignment = np.cos(yaw_error)

        speed_gate = np.clip(speed / 2.0, 0.0, 1.0)
        align_gate = 1.0 if forward_alignment > 0 else 0.0
        
        reward += 10.0 * forward_alignment * speed_gate * align_gate

        # 4. Existential time penalty
        reward -= 1.0

        # 5. Action smoothness
        reward -= 0.5 * action_diff

        if dist < goal_radius:
            return 1000.0 , True
        
        return reward, False
    
    def get_reward2(self, pos, vel, rpy, goal_pos, action_diff, goal_radius=1.0):
        """Calculates the reward based on the current state."""
        dist = float(np.linalg.norm(goal_pos[:2] - pos[:2]))
        speed = float(np.linalg.norm(vel[:2]))

        # 1. THE KILL SWITCH: Terminate on success immediately
        if dist < goal_radius:
            return 1000.0, True

        reward = 0.0

        # 2. Linear distance penalty (Scaled down from 100 to prevent gradient explosions)
        reward -= 1.0 * dist

        # 3. Progress Calculation
        progress = 0.0
        if self.prev_dist is not None:
            progress = self.prev_dist - dist
            safe_progress = np.clip(progress, -2.0, 2.0) 
            reward += 50.0 * safe_progress
        self.prev_dist = dist

        # 4. Heading alignment (TIED TO PROGRESS)
        goal_dir_yaw = np.arctan2(goal_pos[1] - pos[1], goal_pos[0] - pos[0])
        vessel_yaw_rad = np.deg2rad(rpy[2]) 
        yaw_error = abs((goal_dir_yaw - vessel_yaw_rad + np.pi) % (2 * np.pi) - np.pi)
        
        # Linear normalized alignment: 1.0 is perfect aim, 0.0 is facing exactly away
        forward_alignment = (np.pi - yaw_error) / np.pi
        
        speed_gate = np.clip(speed / 2.0, 0.0, 1.0)
        # Gate activates only if pointing forward (error < 90 degrees)
        align_gate = 1.0 if yaw_error < (np.pi / 2.0) else 0.0
        
        # ONLY reward speed and alignment if the vessel is actually getting closer
        if progress > 0:
            reward += 10.0 * forward_alignment * speed_gate * align_gate

        # 5. THE BRAKING ZONE: Penalize speed when very close to the goal
        if dist < 3.0:
            reward -= 5.0 * speed

        # 6. Existential & Smoothness penalties
        reward -= 1.0
        reward -= 0.5 * action_diff
        
        return reward, False
    
    def get_reward3(self, pos, vel, rpy, goal_pos, action, goal_radius=1.0):
        """Calculates the reward based on the current state."""
        dist = float(np.linalg.norm(goal_pos[:2] - pos[:2]))
        speed = float(np.linalg.norm(vel[:2]))

        # 1. THE KILL SWITCH: Terminate on success immediately
        if dist < goal_radius:
            return 1000.0, True

        reward = 0.0

        # 2. Linear distance penalty
        reward -= 10.0 * dist

        # 3. Progress Calculation
        progress = 0.0
        if self.prev_dist is not None:
            progress = self.prev_dist - dist
            safe_progress = np.clip(progress, -2.0, 2.0) 
            reward += 50.0 * safe_progress
        self.prev_dist = dist

        # 4. Heading alignment
        goal_dir_yaw = np.arctan2(goal_pos[1] - pos[1], goal_pos[0] - pos[0])
        vessel_yaw_rad = np.deg2rad(rpy[2]) 
        yaw_error = abs((goal_dir_yaw - vessel_yaw_rad + np.pi) % (2 * np.pi) - np.pi)
        
        forward_alignment = (np.pi - yaw_error) / np.pi
        speed_gate = np.clip(speed / 2.0, 0.0, 1.0)
        align_gate = 1.0 if yaw_error < (np.pi / 2.0) else 0.0
        
        if progress > 0:
            reward += 10.0 * forward_alignment * speed_gate * align_gate

        # --- NEW: Hard Directional Boundary ---
        # Apply a flat penalty if facing more than 90 degrees away from target
        if yaw_error > (np.pi / 2.0):
            reward -= 10.0

        # 5. THE BRAKING ZONE: Penalize speed when very close to the goal
        if dist < 3.0:
            reward -= 5.0 * speed

        # 6. Existential penalty
        reward -= 1.0

        # --- NEW: Advanced Action Smoothness Penalty ---
        self.action_history.append(action)
        if len(self.action_history) > 1:
            # Calculate standard deviation across the history buffer
            sigma_delta = float(np.mean(np.std(self.action_history, axis=0)))
            # Exponential decay penalty: 0 penalty when perfectly smooth, approaching -1 when erratic
            reward += (np.exp(-3.0 * sigma_delta) - 1.0)
        
        return reward, False
    
    def get_rewardS3(self, pos, vel, rpy, goal_pos, action, goal_radius=1.0):  #Smoothed version of get_reward3 with gradual penalties instead of hard cutoffs - WORKSSSSSS
        """Calculates the reward based on the current state."""
        dist = float(np.linalg.norm(goal_pos[:2] - pos[:2]))
        speed = float(np.linalg.norm(vel[:2]))

        # 1. THE KILL SWITCH: Terminate on success immediately
        if dist < goal_radius:
            return 1000.0, True

        reward = 0.0

        # 2. Linear distance penalty (Scaled back to -1.0 to prevent gradient explosions)
        reward -= 1.0 * dist

        # 3. Progress Calculation
        progress = 0.0
        if self.prev_dist is not None:
            progress = self.prev_dist - dist
            safe_progress = np.clip(progress, -2.0, 2.0) 
            reward += 50.0 * safe_progress
        self.prev_dist = dist

        # 4. Heading alignment
        goal_dir_yaw = np.arctan2(goal_pos[1] - pos[1], goal_pos[0] - pos[0])
        vessel_yaw_rad = np.deg2rad(rpy[2]) 
        yaw_error = abs((goal_dir_yaw - vessel_yaw_rad + np.pi) % (2 * np.pi) - np.pi)
        
        forward_alignment = (np.pi - yaw_error) / np.pi
        speed_gate = np.clip(speed / 2.0, 0.0, 1.0)
        align_gate = 1.0 if yaw_error < (np.pi / 2.0) else 0.0
        
        if progress > 0:
            reward += 10.0 * forward_alignment * speed_gate * align_gate

        # --- SMOOTHED: Directional Boundary ---
        # Scales linearly from 0 penalty at 90 degrees to -10 penalty at 180 degrees
        if yaw_error > (np.pi / 2.0):
            # Calculate how far past 90 degrees we are (0.0 to 1.0 scale)
            error_ratio = (yaw_error - (np.pi / 2.0)) / (np.pi / 2.0)
            reward -= 10.0 * error_ratio

        # --- SMOOTHED: Braking Zone ---
        # Scales linearly from 0 at 3.0m, increasing to full penalty at 0.0m
        if dist < 3.0:
            brake_weight = (3.0 - dist) / 3.0
            reward -= 5.0 * speed * brake_weight

        # 6. Existential penalty
        reward -= 1.0

        # 7. Advanced Action Smoothness Penalty 
        self.action_history.append(action)
        if len(self.action_history) > 1:
            sigma_delta = float(np.mean(np.std(self.action_history, axis=0)))
            reward += (np.exp(-3.0 * sigma_delta) - 1.0)
        
        return reward, False

    def get_ca_reward_paper(self, pos, rpy, goal_pos, lidar_distances_raw, current_steering, goal_radius=10.0, dt=0.5):
        """
        Implementation of the Collision Avoidance & Goal Navigation reward 
        from Equations 11-17 in the JMSE paper.
        """
        # 1. R_target (Equation 12): Progress towards goal
        dist_to_goal = float(np.linalg.norm(goal_pos[:2] - pos[:2]))
        R_target = 0.0
        if self.prev_dist is not None:
            R_target = self.prev_dist - dist_to_goal
        self.prev_dist = dist_to_goal

        # 2. R_course' (Equation 11): Alignment with the goal
        goal_yaw_rad = np.arctan2(goal_pos[1] - pos[1], goal_pos[0] - pos[0])
        current_yaw_rad = np.deg2rad(rpy[2])
        # Dot product of heading vector and goal vector
        R_course = np.cos(current_yaw_rad - goal_yaw_rad)

        # 3. R_yaw (Equation 13 adaptation): Penalize erratic steering changes
        R_yaw = -abs(self.prev_steering - current_steering) / dt
        self.prev_steering = current_steering

        # 4. R_collide (Equations 14-16): Weighted LiDAR penalty
        N_l = 64
        max_lidar_range = 70.0
        collision_radius = 4.0  # Increased from 2.0 to account for boat length
        
        # Normalize LiDAR to [0, 1]. 1.0 means no obstacle detected.
        F = np.clip(lidar_distances_raw / max_lidar_range, 0.0, 1.0)
        d_min = np.min(F)
        
        R_collide = 0.0
        if d_min < (collision_radius / max_lidar_range): 
            # Critical collision reached!
            R_collide = -100.0
        elif d_min == 1.0:
            # Clear waters
            R_collide = 0.0
        else:
            # Calculate beam weights (W). Front beams (index 0) have highest weight.
            W = np.zeros(N_l)
            k3 = -0.1
            for i in range(N_l):
                # Assuming index 0 is front, index 32 is rear.
                idx_dist = min(i, N_l - i) 
                W[i] = np.exp(k3 * idx_dist)
            
            # Normalize weights to [0, 1] as per Equation 15
            W = W / np.max(W)
            
            # Calculate penalty (Equation 16): W * (F - 1)
            R_collide = np.sum(W * (F - 1.0))
            
        # 5. Total Reward (Equation 17)
        k4 = 1.0   # Course weight
        k5 = 50.0  # Target progress weight 
        k6 = 0.5   # Steering smoothness weight
        k7 = 5.0   # Collision penalty weight
        
        r_total = k4 * R_course + k5 * R_target + k6 * R_yaw + k7 * R_collide
        
        # Terminal Conditions
        terminated = False
        if dist_to_goal < goal_radius:
            r_total += 1000.0  # Goal reached
            terminated = True
        elif d_min < (collision_radius / max_lidar_range):
            r_total -= 100.0   # Episode failed due to collision
            terminated = True

        return float(r_total), terminated
    
    def _compute_beam_weights(self, N_l=64, k3=-0.1):
        """
        Paper Eq. (14)/(15).
        Beam 0 = front, indices increase clockwise.
        chi_i = angular distance from front beam.
        """
        w = np.zeros(N_l)
        half = N_l // 2

        for i in range(N_l):
            if (N_l % 2 == 0 and i > half) or (N_l % 2 == 1 and i > half):
                chi_i = N_l - i   # back-half: mirror back to [1, half]
            else:
                chi_i = i         # front-half: chi_i = i directly
            w[i] = np.exp(k3 * chi_i)

        # exp(k3 * 0) = 1.0 at front, so max is already 1.0
        # normalise anyway for safety (Eq. 15)
        w = w / np.max(w)
        return w
        
    def get_ca_reward_paper_smoothed(
        self, pos, rpy, goal_pos, lidar_distances_raw, action, vel, yaw_rate,
        a=2.0, b=4.0, k=2.5,
        collision_radius_m=5.0,   # tight realistic physical hitbox for 4m boat
        goal_radius=1.0,
        max_lidar_range=70.0,
        danger_zone_m=5.0,        # Reduced to 5.0m so the repulsion field is tight and accurate
        N_l=64,
        k3=-0.1,
        k4=1.0,    # course alignment
        k5=1.0,    # progress (reduced from 50.0 to prevent farming speed points and crashing)
        k6=1.0,    # action smoothness (reduced from 2.0 to prevent oversaturation)
        k7=0.5,    # collision repulsion (reduced from 10.0 to fix the Suicide Exploit)
        dt=0.5
    ):
        # ------------------------------------------------------------------ #
        # 1. R_target — dense progress reward
        # ------------------------------------------------------------------ #
        dist_to_goal = float(np.linalg.norm(goal_pos[:2] - pos[:2]))
        R_target = 0.0
        if self.prev_dist is not None:
            progress = self.prev_dist - dist_to_goal
            # Asymmetric clip: Cap negative progress (dodging rocks) at -0.5 so it doesn't get violently punished for evasive maneuvers.
            # Leave positive progress capped at 10.0 so it can still achieve high speeds!
            R_target = np.clip(progress, -0.5, 10.0)
        self.prev_dist = dist_to_goal
    
        # ------------------------------------------------------------------ #
        # 2. R_course — heading alignment (penalty for not facing)
        # ------------------------------------------------------------------ #
        goal_yaw_rad    = np.arctan2(goal_pos[1] - pos[1], goal_pos[0] - pos[0])
        current_yaw_rad = np.deg2rad(rpy[2])
        # np.cos is 1.0 when perfectly aligned.
        # This gives a positive signal for correct heading!
        R_course = float(np.cos(current_yaw_rad - goal_yaw_rad))
    
        # ------------------------------------------------------------------ #
        # 3. R_action — action smoothness
        # ------------------------------------------------------------------ #
        self.action_history.append(action)
        R_action = 0.0
        if len(self.action_history) > 1:
            sigma_delta = float(np.mean(np.std(self.action_history, axis=0)))
            R_action = np.exp(-1.0 * sigma_delta) - 1.0
    
        # ------------------------------------------------------------------ #
        # 4. R_collide — paper Eq. (16), linear repulsion field
        # ------------------------------------------------------------------ #
        # Cache weights (computed once, reused every step)
        if self._beam_weights is None:
            self._beam_weights = self._compute_beam_weights(N_l, k3)
        W = self._beam_weights
    
        # Normalise LiDAR to F ∈ [0, 1];  F=1 → clear, F=0 → obstacle at ship
        # Uses danger_zone_m to clip distant threats
        F = np.clip(lidar_distances_raw / danger_zone_m, 0.0, 1.0)
    
        if np.min(F) == 1.0:
            R_collide = 0.0          # no obstacle in any beam → zero penalty
        else:
            # Paper Eq. (16): W^T (F - 1), un-diluted to maintain proper penalty scaling vs progress
            R_collide = float(np.dot(W, F - 1.0))
    
        # ------------------------------------------------------------------ #
        # 5. Continuous total & Spin Penalty
        # ------------------------------------------------------------------ #
        spin_penalty = 2.0 * abs(yaw_rate)
        r_total = k4 * R_course + k5 * R_target + k6 * R_action + k7 * R_collide - spin_penalty - 1.0
    
        # ------------------------------------------------------------------ #
        # 6. Terminal conditions (raw metres, isolated from continuous shaping)
        # ------------------------------------------------------------------ #
        terminated  = False
        term_reward = 0.0
    
        if dist_to_goal < goal_radius:
            r_total     += 1000.0
            term_reward  = 1000.0
            terminated   = True
        elif np.min(lidar_distances_raw) < collision_radius_m:  # raw metres
            r_total     -= 1000.0
            term_reward  = -1000.0
            terminated   = True
    
        # ------------------------------------------------------------------ #
        # 7. Diagnostics
        # ------------------------------------------------------------------ #
        reward_components = {
            "r_course"  : float(k4 * R_course),
            "r_target"  : float(k5 * R_target),
            "r_action"  : float(k6 * R_action),
            "r_collide" : float(k7 * R_collide),
            "r_terminal": float(term_reward),
        }
    
        return float(r_total), terminated, reward_components