import numpy as np
import gymnasium as gym
from gymnasium import spaces
import holoocean
from lidar_scenario import SCENARIO
from reward_minimalist import RewardCalculator
from collections import deque

class SurfaceVesselEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 10}

    def __init__(self, max_steps=1000, scenario_config=SCENARIO, show_viewport=True, goal_radius=1.0, obstacle_spacing=50.0, obstacle_scale=(2.0, 5.0), obstacle_jitter=5.0, moving_target=False, spawn_distance_range=(80.0, 150.0)):
        super().__init__()
        self.max_steps = max_steps
        self.obstacle_spacing = obstacle_spacing
        self.obstacle_scale = obstacle_scale
        self.obstacle_jitter = obstacle_jitter
        self.moving_target = moving_target
        self.spawn_distance_range = spawn_distance_range
        self.step_count = 0
        self.reward_calculator = RewardCalculator()
        self.goal_radius = goal_radius
        self.show_viewport = show_viewport

        # Goal is set to a distant point in the harbor
        self.goal_pos = np.array([278.9, -732.0, 0.0], dtype=np.float32)

        # Observation space: 6 kinematics + 64 LiDAR beams = 70
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(70,), dtype=np.float32
        )

        # Action space: 2D [Throttle, Steering]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        self.show_viewport = show_viewport
        self.sim = None 
        self._last_sensors = {}
        self._prev_action = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Spawn goal randomly based on configured distance range
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(*self.spawn_distance_range)
        self.goal_pos = np.array([distance * np.cos(angle), distance * np.sin(angle), 0.0], dtype=np.float32)
        
        if self.moving_target:
            # Target moves AWAY from spawn at a speed between 0.1 and 0.5 m/s
            speed = np.random.uniform(0.1, 0.5)
            self.goal_vel = np.array([speed * np.cos(angle), speed * np.sin(angle), 0.0], dtype=np.float32)
        else:
            self.goal_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Save initial distance to properly normalize the neural network observations
        self.initial_dist = float(np.linalg.norm(self.goal_pos[:2]))
        
        self.step_count = 0
        self._prev_action = 0.0
        self.reward_calculator.reset()

        self._cleanup_sim()
        self.sim = holoocean.make(scenario_cfg=SCENARIO, show_viewport=self.show_viewport)
        self._spawn_obstacles()

        # Flush initial physics to stabilize and catch first LiDAR frame
        self.sim.act("vessel0", [0.0, 0.0])
        self._last_sensors = None
        for _ in range(50):  
            sensors = self.sim.tick()
            if "Lidar" in sensors and sensors["Lidar"].size > 0:
                self._last_sensors = sensors
                break 
                
        if self._last_sensors is None:
            self._last_sensors = sensors

        if self.show_viewport:
            self._draw_goal_marker()
            dyn = self._last_sensors["DynamicsSensor"]
            self.last_drawn_pos = dyn[6:9].copy()

        obs = self._get_obs(self._last_sensors, current_steering=0.0)
        return obs, self._get_info(self._last_sensors)

    def step(self, action):
        # 1. 2D Action Mapping [Throttle, Steering]
        throttle_action = float(action[0])
        steering_effort = float(action[1])
        
        # Scale throttle from [-1.0, 1.0] to [0.4, 1.0].
        base_pwm = 0.4 + 0.6 * ((throttle_action + 1.0) / 2.0) 
        max_thrust = 5000.0 
        
        left_thrust  = np.clip(max_thrust * (base_pwm + steering_effort), 0.0, max_thrust)
        right_thrust = np.clip(max_thrust * (base_pwm - steering_effort), 0.0, max_thrust)
        self.sim.act("vessel0", [left_thrust, right_thrust])
        
        # 2. Step Physics & Catch Lidar
        latest_valid_lidar = None
        final_sensors = None
        
        for _ in range(10):
            sensors = self.sim.tick()
            if "Lidar" in sensors and sensors["Lidar"].size > 0:
                latest_valid_lidar = sensors["Lidar"].copy()
            final_sensors = sensors
            
        if latest_valid_lidar is not None:
            final_sensors["Lidar"] = latest_valid_lidar
        else:
            final_sensors["Lidar"] = self._last_sensors.get("Lidar", np.array([]))
            
        self._last_sensors = final_sensors

        # 3. Calculate State & Reward
        obs = self._get_obs(final_sensors, steering_effort)
        
        # Extract components for reward
        dyn = final_sensors["DynamicsSensor"]
        pos = dyn[6:9]
        rpy = dyn[15:18].astype(np.float32)
        
        current_yaw_rad = np.deg2rad(rpy[2])
        goal_yaw_rad = np.arctan2(self.goal_pos[1] - pos[1], self.goal_pos[0] - pos[0])
        yaw_error = (goal_yaw_rad - current_yaw_rad + np.pi) % (2 * np.pi) - np.pi
        
        lidar_distances_raw = obs[6:] * 70.0 # Un-normalize LiDAR for reward logic

        # --- PURSUIT EVASION: Move the target ---
        dt = 0.5
        self.goal_pos += self.goal_vel * dt

        if self.show_viewport:
            self._draw_trajectory(pos)
            self._draw_goal_marker()

        r_total, terminated, reward_components = self.reward_calculator.get_minimalist_reward(
            pos, self.goal_pos, lidar_distances_raw, yaw_error, goal_radius=self.goal_radius
        )
        
        self._prev_action = steering_effort
        self.step_count += 1
        truncated = bool(self.step_count >= self.max_steps)
        
        info = self._get_info(final_sensors)
        info.update(reward_components)
        
        return obs, r_total, terminated, truncated, info

    def _get_obs(self, sensors, current_steering):
        dyn = sensors["DynamicsSensor"]
        pos = dyn[6:9].astype(np.float32)
        rpy = dyn[15:18].astype(np.float32)
        yaw_rate = dyn[14]

        current_yaw_rad = np.deg2rad(rpy[2])
        goal_yaw_rad = np.arctan2(self.goal_pos[1] - pos[1], self.goal_pos[0] - pos[0])
        dist_to_goal = np.linalg.norm(self.goal_pos[:2] - pos[:2])
        speed = np.linalg.norm(dyn[3:5])

        # --- 1. Kinematics (6 Dimensions) ---
        # Features: [normalized_dist, sin(yaw_error), cos(yaw_error), normalized_speed, steering, yaw_rate]
        yaw_error = (goal_yaw_rad - current_yaw_rad + np.pi) % (2 * np.pi) - np.pi
        
        # Scale-invariant distance: starts at 1.0, goes to 0.0. Clamped to 1.0 so PE doesn't break OOD.
        normalized_dist = np.clip(dist_to_goal / getattr(self, 'initial_dist', 565.0), 0.0, 1.0)
        
        kinematics = np.array([
            normalized_dist,
            np.sin(yaw_error),
            np.cos(yaw_error),
            speed / 10.0,
            current_steering,
            np.clip(yaw_rate, -1.0, 1.0)
        ], dtype=np.float32)

        # --- 2. Perception (64 Dimensions) ---
        lidar_distances = np.full(64, 70.0, dtype=np.float32)
        
        if "Lidar" in sensors and sensors["Lidar"].size > 0:
            lidar_data = sensors["Lidar"] 
            if lidar_data.ndim == 1:
                coords_per_point = 4 if len(lidar_data) % 4 == 0 else 3
                lidar_data = lidar_data.reshape(-1, coords_per_point)
            
            points_xyz = lidar_data[:, :3]
            hit_distances = np.linalg.norm(points_xyz, axis=1)
            
            # Simple angle binning without DBSCAN clustering overhead
            angles = np.arctan2(points_xyz[:, 1], points_xyz[:, 0]) 
            angles_deg = (np.degrees(angles) + 360) % 360           
            beam_indices = np.floor(angles_deg / 5.625).astype(int)
            beam_indices = np.clip(beam_indices, 0, 63) 
            
            for idx, beam_dist in zip(beam_indices, hit_distances):
                if beam_dist < lidar_distances[idx]:
                    lidar_distances[idx] = beam_dist
                    
        normalized_lidar = lidar_distances / 70.0
        
        return np.concatenate([kinematics, normalized_lidar], dtype=np.float32)

    def _get_info(self, sensors):
        dyn = sensors["DynamicsSensor"]
        pos = dyn[6:9]
        vel = dyn[3:6]
        return {
            "dist_to_goal": float(np.linalg.norm(self.goal_pos[:2] - pos[:2])),
            "speed": float(np.linalg.norm(vel[:2])),
            "step_count": self.step_count,
            "pos_x": float(pos[0]),
            "pos_y": float(pos[1]),
            "goal_x": float(self.goal_pos[0]),
            "goal_y": float(self.goal_pos[1])
        }

    '''
    def _spawn_obstacles(self, n=3000):
        # The center of your obstacle field
        spawn = np.array([0.0, 0.0]) 
        
        shapes = ["cylinder", "box", "sphere"]
        
        # Define the 360-degree zone boundaries
        inner_radius = 20.0   # Keeps a 20m safe bubble around the boat at startup
        outer_radius = 600.0  # Scatters them up to 350m away in all directions
        
        for _ in range(n):
            # 1. Pick a random angle (0 to 2*pi radians)
            angle = np.random.uniform(0, 2 * np.pi)
            
            # 2. Pick a random distance from the spawn
            # (Using sqrt ensures they are evenly distributed across the area, not clumped in the center)
            radius = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2))
            
            # 3. Convert Polar Coordinates (angle & radius) back to X,Y map coordinates
            obs_x = spawn[0] + radius * np.cos(angle)
            obs_y = spawn[1] + radius * np.sin(angle)
            
            # Randomize shape and size
            chosen_shape = np.random.choice(shapes)
            scale_x = np.random.uniform(2.0, 6.0) 
            scale_y = np.random.uniform(2.0, 6.0) 
            
            self.sim.spawn_prop(
                prop_type=chosen_shape,
                location=[float(obs_x), float(obs_y), 0.2],
                scale=[scale_x, scale_y, 6.0], 
                sim_physics=False,
                material="wood"
            )
        '''
    
    def _spawn_obstacles(self):
        self.obstacle_coords = []
        # Get spawn and goal positions to center the grid
        spawn = np.array([0.0, 0.0])
        center_x = (spawn[0] + self.goal_pos[0]) / 2.0
        center_y = (spawn[1] + self.goal_pos[1]) / 2.0
        
        shapes = ["cylinder", "box", "sphere"]
        
        # --- Grid Configuration ---
        # 1.2km by 1.2km grid centered directly over the path!
        grid_width = 1200.0  
        grid_height = 1200.0 
        spacing = self.obstacle_spacing
        
        # Create structured arrays for X and Y coordinates relative to center
        x_coords = np.arange(center_x - grid_width/2, center_x + grid_width/2, spacing)
        y_coords = np.arange(center_y - grid_height/2, center_y + grid_height/2, spacing)
        
        for i, y in enumerate(y_coords):
            # Stagger every other row by half the spacing to create a zig-zag / honeycomb checkerboard
            offset_x = (spacing / 2.0) if i % 2 != 0 else 0.0
            
            for x in x_coords:
                # Apply the zig-zag offset
                staggered_x = x + offset_x
                
                # 1. The Safe Bubble: Don't spawn inside the starting area
                dist_to_spawn = np.linalg.norm([staggered_x, y])
                if dist_to_spawn < 20.0:
                    continue # Skip this grid point
                
                # 2. Add Massive "Jitter" (Destroys straight lanes)
                # By increasing the jitter to +/- self.obstacle_jitter (nearly half the spacing), 
                # we completely shatter the honeycomb grid into a chaotic, organic minefield.
                jitter_x = np.random.uniform(-self.obstacle_jitter, self.obstacle_jitter)
                jitter_y = np.random.uniform(-self.obstacle_jitter, self.obstacle_jitter)
                
                obs_x = spawn[0] + staggered_x + jitter_x
                obs_y = spawn[1] + y + jitter_y
                self.obstacle_coords.append((float(obs_x), float(obs_y)))
                
                # 3. Randomize shape and size
                chosen_shape = np.random.choice(shapes)
                scale_x = np.random.uniform(self.obstacle_scale[0], self.obstacle_scale[1]) 
                scale_y = np.random.uniform(self.obstacle_scale[0], self.obstacle_scale[1]) 
                
                self.sim.spawn_prop(
                    prop_type=chosen_shape,
                    location=[float(obs_x), float(obs_y), 0.2],
                    scale=[scale_x, scale_y, 6.0], 
                    sim_physics=False,
                    material="wood"
                )

    def _draw_goal_marker(self):
        gx, gy = float(self.goal_pos[0]), float(self.goal_pos[1])
        z = 0.2 # Raised slightly so it doesn't clip under the water graphics

        # Big box around the goal area (Green)
        self.sim.draw_box(
            center=[gx, gy, z],
            extent=[self.goal_radius, self.goal_radius, 1.0],
            color=[0, 255, 0],
            thickness=3.0,
            lifetime=0.6
        )

        # Bright point at the exact goal centre (Red)
        self.sim.draw_point(
            loc=[gx, gy, z],
            color=[255, 0, 0],
            thickness=15.0,
            lifetime=0.6
        )

    def _draw_trajectory(self, current_pos):
        if hasattr(self, 'last_drawn_pos') and self.last_drawn_pos is not None:
            # Only draw a new line segment if the boat has moved more than 0.5 meters
            # This prevents memory lag from drawing thousands of tiny overlapping lines
            if np.linalg.norm(current_pos[:2] - self.last_drawn_pos[:2]) > 0.5:
                self.sim.draw_line(
                    start=[float(self.last_drawn_pos[0]), float(self.last_drawn_pos[1]), 0.2],
                    end=[float(current_pos[0]), float(current_pos[1]), 0.2],
                    color=[100, 200, 255], # Bright Blue trail
                    thickness=5.0,
                    lifetime=0 # Persist forever
                )
                self.last_drawn_pos = current_pos.copy()

    def _cleanup_sim(self):
        if self.sim is not None:
            sim = self.sim
            self.sim = None

            try:
                if hasattr(sim, "_client") and hasattr(sim._client, "unlink"):
                    sim._client.unlink()
            except Exception:
                pass

            try:
                if hasattr(sim, "_world_process"):
                    sim._world_process.kill()
                    sim._world_process.wait(10)
            except Exception:
                pass

            try:
                sim.agents.clear()
                sim._state_dict.clear()
                sim._client._memory.clear()
                sim._exited = True
            except Exception:
                pass
            
        import gc
        import os
        
        # Force garbage collection to ensure HoloOcean's Shmem __del__ methods are called
        # which will os.close() their file descriptors. Without this, circular references 
        # keep the FDs open and cause "OSError: Too many open files".
        gc.collect()

    def close(self):
        self._cleanup_sim()