import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_rewards(log_path, output_img):
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    episodes = []
    returns = []
    distances = []
    
    # EP 83 | env=0 | step=51797 | return=924.79 | len=967 | dist=0.92
    pattern = re.compile(r'EP (\d+) \| env=\d+ \| step=(\d+) \| return=([-+]?\d*\.\d+|\d+) \| len=(\d+) \| dist=([-+]?\d*\.\d+|\d+)')
    
    for line in lines:
        match = pattern.search(line)
        if match:
            ep = int(match.group(1))
            r_total = float(match.group(3))
            dist = float(match.group(5))
            episodes.append(ep)
            returns.append(r_total)
            distances.append(dist)
            
    if not episodes:
        print("No data found!")
        return
        
    df = pd.DataFrame({'Episode': episodes, 'Return': returns, 'Distance': distances})
    
    # Smoothing using a rolling window
    df['Smoothed Return (MA=5)'] = df['Return'].rolling(window=5, min_periods=1).mean()
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Scatter plot of all returns
    plt.scatter(df['Episode'], df['Return'], color='skyblue', alpha=0.6, label='Raw Episodic Return')
    
    # Line plot of smoothed returns
    plt.plot(df['Episode'], df['Smoothed Return (MA=5)'], color='blue', linewidth=2, label='Smoothed Return (Trend)')
    
    # Highlight successful episodes (dist < 1.0)
    successes = df[df['Distance'] < 1.0]
    plt.scatter(successes['Episode'], successes['Return'], color='gold', edgecolor='black', s=100, zorder=5, label='Goal Reached (+1000 Bonus)')
    
    # Add a horizontal dashed line at 0 for reference
    plt.axhline(0, color='red', linestyle='--', alpha=0.8, label='Zero Threshold')
    
    plt.title('SAC Training: Episodic Returns over Time (0 - 70k Steps)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Return)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    print(f"Saved plot to {output_img}")

if __name__ == "__main__":
    log_file = '/home/trizzz/.gemini/antigravity/brain/4c88dfe3-247d-4644-a441-b1035059495f/.system_generated/tasks/task-2329.log'
    out_file = '/home/trizzz/.gemini/antigravity/brain/4c88dfe3-247d-4644-a441-b1035059495f/episodic_return_70k.png'
    plot_rewards(log_file, out_file)
