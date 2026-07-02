import pandas as pd
import glob
import os

def analyze_latest_run():
    # Find the most recent run directory
    run_dirs = glob.glob("runs/SurfaceVessel-v0__cleanrl_sac__1__*")
    if not run_dirs:
        print("No run directories found.")
        return
        
    latest_run = max(run_dirs, key=os.path.getmtime)
    csv_path = os.path.join(latest_run, "reward_components.csv")
    
    if not os.path.exists(csv_path):
        print(f"CSV not found in latest run: {csv_path}")
        return
        
    print(f"Analyzing {csv_path}...\n")
    df = pd.read_csv(csv_path)
    
    # Calculate statistics for each component
    components = ['r_total', 'r_course', 'r_target', 'r_action', 'r_collide', 'r_terminal']
    
    # We want to see how much each component contributes on average (absolute magnitude)
    print("--- Summary Statistics ---")
    print(df[components].describe().to_string())
    print("\n--- Mean Absolute Contribution ---")
    for comp in components:
        mean_abs = df[comp].abs().mean()
        print(f"{comp}: {mean_abs:.4f}")
        
    print("\n--- Contribution % (based on sum of mean absolute values) ---")
    total_abs_mean = sum(df[comp].abs().mean() for comp in components if comp != 'r_total')
    if total_abs_mean > 0:
        for comp in components:
            if comp != 'r_total':
                perc = (df[comp].abs().mean() / total_abs_mean) * 100
                print(f"{comp}: {perc:.2f}%")
                
    print("\nTotal Steps Processed:", len(df))

if __name__ == "__main__":
    analyze_latest_run()
