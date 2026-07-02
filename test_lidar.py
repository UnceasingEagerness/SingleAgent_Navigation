import holoocean
import numpy as np
from pynput import keyboard

scenario = {
    "name": "find_piers",
    "package_name": "Ocean",
    "world": "PierHarbor",
    "main_agent": "vessel0",
    "agents": [
        {
            "agent_name": "vessel0",
            "agent_type": "SurfaceVessel",
            "sensors": [
                {"sensor_type": "DynamicsSensor", "sensor_name": "DynamicsSensor"}
            ],
            "control_scheme": 0,
            "location": [-236.9, -697.5, 0.2],
            "rotation": [0, 0, 0]
        }
    ]
}

def spawn_obstacles(env, randomize=False, n=6):
    spawn = np.array([-236.9, -697.5])

    if randomize:
        for _ in range(n):
            while True:
                offset = np.random.uniform(-30, 30, size=2)
                if np.linalg.norm(offset) > 4.0:
                    break
            pos = spawn + offset
            env.spawn_prop(
                prop_type="cylinder",
                location=[float(pos[0]), float(pos[1]), 0.2],
                scale=[2.0, 2.0, 6.0],
                sim_physics=False,
                material="steel"
            )
    else:
        fixed_offsets = [
            [ 10.0,   0.0],
            [  8.0,   8.0],
            [  0.0,  10.0],
            [ 10.0,  -8.0],
            [ 20.0,   5.0],
            [ 15.0, -12.0],
        ]
        for offset in fixed_offsets:
            pos = spawn + np.array(offset)
            env.spawn_prop(
                prop_type="cylinder",
                location=[float(pos[0]), float(pos[1]), 0.2],
                scale=[2.0, 2.0, 6.0],
                sim_physics=False,
                material="steel"
            )

pressed = set()

def on_press(key):
    try:
        pressed.add(key.char)
    except AttributeError:
        pass

def on_release(key):
    try:
        pressed.discard(key.char)
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

with holoocean.make(scenario_cfg=scenario, show_viewport=True) as env:
    spawn_obstacles(env)
    print("WASD to drive | Q to quit")

    while True:
        if 'q' in pressed:
            break

        left, right = 0.0, 0.0
        if 'w' in pressed:
            left, right = 500.0, 500.0
        if 's' in pressed:
            left, right = -500.0, -500.0
        if 'a' in pressed:
            left, right = -300.0, 300.0
        if 'd' in pressed:
            left, right = 300.0, -300.0

        sensors = env.step(np.array([left, right]))
        pos = sensors["DynamicsSensor"][6:9]
        print(f"pos: x={pos[0]:.1f}  y={pos[1]:.1f}  z={pos[2]:.1f}", end="\r")

listener.stop()