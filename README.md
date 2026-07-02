# Single Agent Collision Avoidance

This directory contains the code for training a reinforcement learning agent for path planning and collision avoidance using SAC (Soft Actor-Critic) in the HoloOcean simulator.

## Running the Code

To train the agent, run the following command:
```bash
python3 cleanrl_sac.py
```

## Docker Troubleshooting

If you are running this environment inside a Docker container, you might encounter issues with the HoloOcean simulator starting up properly.

### Issue: `posix_ipc.BusyError: Semaphore is busy` or `HoloOceanException: Timed out waiting for binary to load`

This error commonly occurs when a previous run crashed or was killed abruptly (e.g., using `Ctrl+C`), leaving behind orphaned POSIX semaphores or hanging Unreal Engine processes in the background.

**Solution: Clean up stalled state**

Run the following commands inside your Docker container to kill any hanging processes and clear the orphaned semaphores:

```bash
# Kill any hanging Unreal Engine processes
pkill -9 -f LinuxNoEditor
pkill -9 -f holoocean

# Clear orphaned POSIX semaphores in shared memory
rm -rf /dev/shm/sem.*
```

### Issue: Simulator crashes on startup (Missing Display Server)

If you have cleaned up the semaphores and processes but still receive the same timeout error, it is likely because the Unreal Engine binary cannot find an X11 display server or rendering API (like Vulkan/OpenGL) inside your Docker container.

**Solution: Run with `xvfb` (X virtual framebuffer)**

You can run the script headlessly using `xvfb`. First, ensure it is installed, then wrap your command with `xvfb-run`:

```bash
# Install xvfb if not already installed
sudo apt-get update && sudo apt-get install -y xvfb

# Run the training script headlessly
xvfb-run -a python3 cleanrl_sac.py
```

*Note: The warning about "Ensure that holoocean is not being run with root privileges" is a generic fallback message. If your user ID inside the container is not 0 (you can check with `id`), you can safely ignore that specific warning and focus on the display/semaphore solutions above.*
