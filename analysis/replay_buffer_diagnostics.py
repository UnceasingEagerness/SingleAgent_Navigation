import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

def run_replay_analysis():
    print("--- Replay Buffer Science (Level 7) ---")
    
    runs_dir = '/home/trizzz/AUV_Project/IISC_Project/SurfaceVessel/SAC-PathPlanning&CollisionAvoidance/SingleAgent_CollisionAvoidance/runs'
    all_runs = glob.glob(os.path.join(runs_dir, '*'))
    
    best_csv = None
    max_len = 0
    
    for r in all_runs:
        csv_path = os.path.join(r, 'reward_components.csv')
        if os.path.exists(csv_path):
            df_len = sum(1 for line in open(csv_path))
            if df_len > max_len:
                max_len = df_len
                best_csv = csv_path
                
    if not best_csv:
        print("No CSV found.")
        return
        
    print(f"Loading data from {best_csv}")
    df = pd.read_csv(best_csv)
    
    terminals = df[df['r_terminal'] != 0.0]
    
    # Using the threshold logic because of old CSV format
    successes = terminals[terminals['r_terminal'] > 500.0]
    crashes = terminals[terminals['r_terminal'] < -500.0]
    
    total_episodes = len(terminals)
    
    if total_episodes == 0:
        print("No episodes found.")
        return
        
    success_ratio = len(successes) / total_episodes * 100.0
    crash_ratio = len(crashes) / total_episodes * 100.0
    timeout_ratio = (total_episodes - len(successes) - len(crashes)) / total_episodes * 100.0
    
    labels = ['Goal Reached (Success)', 'Crashes', 'Timeouts (Max Steps)']
    sizes = [success_ratio, crash_ratio, timeout_ratio]
    colors = ['#2ca02c', '#d62728', '#1f77b4']
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, shadow=True, explode=(0.05, 0.05, 0.05))
    plt.title('Replay Buffer Experience Distribution (What is the agent eating?)', fontsize=16)
    
    output_path = '/home/trizzz/.gemini/antigravity/brain/4c88dfe3-247d-4644-a441-b1035059495f/replay_buffer_diagnostics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved Replay Buffer pie chart to {output_path}")

if __name__ == "__main__":
    run_replay_analysis()
