import holoocean
import numpy as np

# We'll just define a minimal scenario with 1 boat and 1 obstacle directly in front (+X)
scenario = {
    "name": "LidarTest",
    "package_name": "Ocean",
    "world": "OpenWater",
    "main_agent": "vessel0",
    "agents": [
        {
            "agent_name": "vessel0",
            "agent_type": "SurfaceVessel",
            "sensors": [
                {
                    "sensor_type": "RaycastLidar",
                    "sensor_name": "Lidar",
                    "Hz": 10,
                    "location": [0.0, 0.0, 1.5],
                    "configuration": {
                        "Channels": 1,
                        "Range": 70.0,
                        "HorizontalFov": 360.0,
                        "PointsPerSecond": 640,
                        "RotationFrequency": 10
                    }
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, 0],
            "rotation": [0, 0, 0]
        },
        {
            "agent_name": "obstacle_x",
            "agent_type": "Prop",
            "is_obstacle": True,
            "sensors": [],
            "location": [10, 0, 1], # Directly in front on the X axis
            "scale": [1, 1, 10]
        },
        {
            "agent_name": "obstacle_y",
            "agent_type": "Prop",
            "is_obstacle": True,
            "sensors": [],
            "location": [0, 10, 1], # Directly to the side on the Y axis
            "scale": [1, 1, 10]
        }
    ]
}

print("Spawning HoloOcean Lidar test...")
env = holoocean.make(scenario_cfg=scenario, show_viewport=False)
env.reset()

for i in range(10):
    env.step([0.0, 0.0]) # don't move
    
state = env.tick()
lidar = state['Lidar']
distances = np.linalg.norm(lidar, axis=-1)

# We expect 64 beams (640 pts/sec / 10 Hz)
# Let's find which beam index detected the obstacle at X (should be ~10m)
# and which detected Y (should be ~10m)
min_idx = np.argmin(distances)
min_dist = distances[min_idx]

print(f"Total beams: {len(distances)}")
for i, d in enumerate(distances):
    if d < 12.0:
        print(f"Beam {i} detected obstacle at {d:.2f} meters.")

env.close()
