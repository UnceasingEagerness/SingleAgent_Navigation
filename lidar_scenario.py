# SV_scenario.py
SCENARIO = {
    "name": "SurfaceVesselNav",
    "package_name": "Ocean",
    #"world": "PierHarbor",
    "world": "OpenWater",  # change to a world with water surface
    "main_agent": "vessel0",
    "show_viewport": False,
    "window_width": 256,
    "window_height": 256,
    "agents": [
        {
            "agent_name": "vessel0",
            "agent_type": "SurfaceVessel",
            "sensors": [
                {
                    "sensor_type": "DynamicsSensor",
                    "sensor_name": "DynamicsSensor"
                },
                {
                    "sensor_type": "RaycastLidar",
                    "sensor_name": "Lidar",
                    "Hz": 10,
                    "location": [0.0, 0.0, 1.5],
                    "configuration": {
                        "Channels": 1,
                        "Range": 70.0,
                        "HorizontalFov": 360.0,
                        "RotationFrequency": 10,
                        "PointsPerSecond": 1000,
                        "UpperFovLimit": 0.0,
                        "LowerFovLimit": 0.0,
                        "ShowDebugPoints": False,
                        "NoiseStdDev": 0.0,
                        "AtmospAttenRate": 0.0,
                        "DropOffGenRate": 0.0,
                        "DropOffIntensityLimit": 0.0,
                        "DropOffAtZeroIntensity": 0.0
                    }
                }
                # SinglebeamSonar removed — Lidar replaces it in _get_obs already
            ],
            "control_scheme": 0,
            "location": [0, 0, 0],
            #"location": [278.9, -732.0, 0],
            #"location": [-236.9, -697.5, 0.2],  
            "rotation": [0, 0, 0]
        }
           # unpack all obstacle agents into the agents list
    ],
    "ticks_per_sec": 200,
    "frames_per_sec": True
}