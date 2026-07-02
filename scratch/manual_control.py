import pygame
import numpy as np
import sys
import os

# Ensure the root directory is in the path so we can import surfacevessel_env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())

from surfacevessel_env import SurfaceVesselEnv

def main():
    print("Initializing Pygame and Environment...")
    pygame.init()
    
    # Create a tiny pygame window just to capture keyboard focus
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Holoocean WASD Control (Click here to focus!)")
    
    font = pygame.font.SysFont(None, 24)
    
    # Initialize Environment on Phase 3 settings
    # We set max_steps huge so you don't get kicked out while exploring
    print("Spawning Holoocean Viewport...")
    env = SurfaceVesselEnv(
        max_steps=10000, 
        show_viewport=True, 
        obstacle_spacing=30.0, 
        # Using jitter=12.0 for the chaotic Phase 3 map!
    )
    
    # We will hack the jitter internally since it's hardcoded in the class in this snapshot
    env.reset()
    
    print("\n" + "="*50)
    print("MANUAL CONTROL READY")
    print("="*50)
    print("CLICK on the small Pygame window to capture your keyboard!")
    print("Controls:")
    print("  W : Accelerate (Throttle +)")
    print("  S : Decelerate (Throttle -)")
    print("  A : Steer Left")
    print("  D : Steer Right")
    print("  ESC : Quit")
    print("="*50 + "\n")
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        throttle = -1.0 # Default to idle/slow
        steering = 0.0
        
        # Event pump
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False
            
        if keys[pygame.K_w]:
            throttle = 1.0  # Max throttle
        elif keys[pygame.K_s]:
            throttle = -1.0 # Min throttle
            
        if keys[pygame.K_a]:
            steering = 1.0  # Steer left (positive)
        elif keys[pygame.K_d]:
            steering = -1.0 # Steer right (negative)
            
        # Step the environment
        action = np.array([throttle, steering], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract the LiDAR from the observation
        # According to the wrapper, the first 6 are kinematics, the next 64 are LiDAR
        lidar_scans = obs[6:70]
        min_lidar_dist = np.min(lidar_scans)
        dist_to_goal = info.get("dist_to_goal", 0.0)
        
        # Print to terminal
        print(f"\rThrottle: {throttle:5.1f} | Steering: {steering:5.1f} | Min LiDAR Dist: {min_lidar_dist:6.2f}m | Dist to Goal: {dist_to_goal:6.2f}m", end="")
        
        # Update Pygame window
        screen.fill((0, 0, 0))
        text1 = font.render(f"Min LiDAR Dist: {min_lidar_dist:.2f}m", True, (255, 255, 255))
        text2 = font.render(f"Dist to Goal: {dist_to_goal:.2f}m", True, (255, 255, 255))
        text3 = font.render("Use W/A/S/D to drive!", True, (0, 255, 0))
        screen.blit(text1, (20, 50))
        screen.blit(text2, (20, 100))
        screen.blit(text3, (20, 150))
        pygame.display.flip()
        
        # Run at ~30 FPS
        clock.tick(30)
        
        # We intentionally ignore the 'terminated' flag here so you can keep driving
        # through rocks without it resetting the simulation!
        if truncated:
            print("\nMax steps reached. Respawning...")
            env.reset()

    env.close()
    pygame.quit()
    print("\nExiting Manual Control.")

if __name__ == "__main__":
    main()
