import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

def analyze_success_rate():
    print("--- Success Rate Analysis ---")
    
    # 1. Find the 300k run from yesterday
    runs_dir = '/home/trizzz/AUV_Project/IISC_Project/SurfaceVessel/SAC-PathPlanning&CollisionAvoidance/SingleAgent_CollisionAvoidance/runs'
    # The 300k run from yesterday is likely 1782673464, but we can just sort by size of the CSV
    all_runs = glob.glob(os.path.join(runs_dir, '*'))
    
    best_csv = None
    max_len = 0
    
    for r in all_runs:
        csv_path = os.path.join(r, 'reward_components.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if len(df) > max_len:
                max_len = len(df)
                best_csv = csv_path
                
    if not best_csv:
        print("No CSV found.")
        return
        
    print(f"Loading data from largest run: {best_csv}")
    print(f"Total Steps: {max_len}")
    
    df = pd.read_csv(best_csv)
    
    # Extract episodes
    # An episode ends when r_terminal is not 0
    terminals = df[df['r_terminal'] != 0.0].copy()
    
    if len(terminals) == 0:
        print("No terminal states found in this run!")
        return
        
    # In the old 300k run, the logger recorded the entire step reward as r_terminal
    # A goal catch gives +1000, so total reward is usually > 500
    # A crash gives -1000, so total reward is usually < -500
    terminals['is_success'] = terminals['r_terminal'] > 500.0
    terminals['is_crash'] = terminals['r_terminal'] < -500.0
    
    # Let's calculate a rolling success rate over windows of 50 episodes
    window_size = 50
    if len(terminals) < window_size:
        window_size = max(1, len(terminals) // 2)
        
    terminals['rolling_success'] = terminals['is_success'].rolling(window=window_size).mean() * 100.0
    terminals['rolling_crash'] = terminals['is_crash'].rolling(window=window_size).mean() * 100.0
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(terminals['step'], terminals['rolling_success'], label='Success Rate (%)', color='green', linewidth=2)
    plt.plot(terminals['step'], terminals['rolling_crash'], label='Crash Rate (%)', color='red', linewidth=2)
    
    plt.title(f'Agent Performance Over Time (Rolling {window_size} Episodes)', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = '/home/trizzz/.gemini/antigravity/brain/4c88dfe3-247d-4644-a441-b1035059495f/success_rate.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved Success Rate graph to {output_path}")
    
    print("\n--- Final Performance Metrics ---")
    recent = terminals.tail(100)
    print(f"Last 100 Episodes Success Rate: {recent['is_success'].mean()*100:.1f}%")
    print(f"Last 100 Episodes Crash Rate: {recent['is_crash'].mean()*100:.1f}%")

if __name__ == "__main__":
    analyze_success_rate()
