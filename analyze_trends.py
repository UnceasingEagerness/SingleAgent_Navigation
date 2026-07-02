import pandas as pd
import numpy as np

# Load latest run's CSV
import glob
import os
list_of_files = glob.glob('runs/*/reward_components.csv')
latest_file = max(list_of_files, key=os.path.getctime)

df = pd.read_csv(latest_file)

# Episodic Returns
term_indices = df.index[df['r_terminal'] != 0.0].tolist()
ep_returns = []
start_idx = 0
for idx in term_indices:
    ep_returns.append(df['r_total'].iloc[start_idx:idx+1].sum())
    start_idx = idx + 1

print(f"Total Episodes Completed: {len(ep_returns)}")

if len(ep_returns) > 10:
    first_10 = np.mean(ep_returns[:10])
    last_10 = np.mean(ep_returns[-10:])
    print(f"Average Return (First 10): {first_10:.2f}")
    print(f"Average Return (Last 10): {last_10:.2f}")
    
    # Also check collisions vs goals
    df_terminals = df.loc[term_indices]
    goals = len(df_terminals[df_terminals['r_terminal'] > 0])
    crashes = len(df_terminals[df_terminals['r_terminal'] < 0])
    print(f"Total Goals Reached (+1000): {goals}")
    print(f"Total Crashes (-100): {crashes}")
    
    # Goal/Crash ratio in first 20% vs last 20%
    n_20 = max(1, int(len(df_terminals) * 0.2))
    first_20_term = df_terminals.iloc[:n_20]
    last_20_term = df_terminals.iloc[-n_20:]
    
    print(f"First 20% -> Goals: {len(first_20_term[first_20_term['r_terminal']>0])}, Crashes: {len(first_20_term[first_20_term['r_terminal']<0])}")
    print(f"Last 20% -> Goals: {len(last_20_term[last_20_term['r_terminal']>0])}, Crashes: {len(last_20_term[last_20_term['r_terminal']<0])}")
