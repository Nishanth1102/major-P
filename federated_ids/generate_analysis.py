import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    base_file = "results/baseline_metrics.csv"
    comp_file = "results/compressed_metrics.csv"
    log_file  = "results/client0_log_compressed.txt"

    if not os.path.exists(base_file) or not os.path.exists(comp_file):
        print("Metrics files not found in 'results/'. Please run `python3 run_experiments.py` first.")
        return

    # Load accuracy data
    df_base = pd.read_csv(base_file)
    df_comp = pd.read_csv(comp_file)

    # 1. Extract Bandwidth/Compression stats from the client log
    orig_kb = 0.0
    comp_kb = 0.0
    reduction = 0.0
    
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            for line in f:
                if "Original :" in line and "KB" in line:
                    orig_kb = float(line.split(":")[1].split("KB")[0].strip())
                if "Compressed:" in line and "KB" in line:
                    comp_kb = float(line.split(":")[1].split("KB")[0].strip())
                if "Reduction :" in line and "%" in line:
                    reduction = float(line.split(":")[1].split("%")[0].strip())
                    break # We only need to read the first round's calculation

    # 2. Print Detailed Analysis parameters for the research paper
    print("\n" + "="*60)
    print(" COMMUNICATION BANDWIDTH & PARAMETER ANALYSIS ")
    print("="*60)
    
    if orig_kb > 0:
        rounds = len(df_base) if "round" in df_base else 10
        clients = 2
        
        print(f"Model Baseline Upload (Per Client/Round) : {orig_kb:8.2f} KB (100.0%)")
        print(f"Model Compressed Upload (Per Client/Round): {comp_kb:8.2f} KB")
        print(f"Achieved Payload Reduction                : {reduction:8.2f} %\n")
        
        print(f"Total FL Rounds executed                  : {rounds}")
        print(f"Total Federated Clients                   : {clients}")
        
        total_base = orig_kb * rounds * clients
        total_comp = comp_kb * rounds * clients
        
        print("\n--- Total Traffic Simulated (Uploads) ---")
        print(f"Standard FedAvg (Uncompressed)            : {total_base / 1024:.2f} MB")
        print(f"Proposed Pipeline (Compressed)            : {total_comp / 1024:.2f} MB")
        print(f"Total Bandwidth Saved by Edge Devices     : {(total_base - total_comp) / 1024:.2f} MB")
    else:
        print("WARNING: Could not parse bandwidth metrics from the compressed client log.")
        print("Ensure 'USE_COMPRESSION=True' logged compression stats correctly.")

    print("\n" + "="*60)
    print(" ACCURACY PERFORMANCE ANALYSIS ")
    print("="*60)
    final_acc_base = df_base['accuracy'].iloc[-1] * 100
    final_acc_comp = df_comp['accuracy'].iloc[-1] * 100
    print(f"Final Global Accuracy (Baseline)   : {final_acc_base:.2f} %")
    print(f"Final Global Accuracy (Compressed) : {final_acc_comp:.2f} %")
    print(f"Accuracy Trade-off Delta           : {abs(final_acc_base - final_acc_comp):.2f} %")

    # 3. Generate Comparative Plot (Dual Subplot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Convergence Analysis
    ax1.plot(df_base['round'], df_base['accuracy'] * 100, marker='o', linewidth=2.5,
             markersize=8, color='crimson', alpha=0.9, label='Baseline (FedAvg)')
             
    ax1.plot(df_comp['round'], df_comp['accuracy'] * 100, marker='s', linewidth=2.5,
             markersize=8, color='forestgreen', alpha=0.9, label='Proposed (Top-K + FP16)')
             
    ax1.set_xlabel('FL Round', fontsize=13)
    ax1.set_ylabel('Global Test Accuracy (%)', fontsize=13)
    ax1.set_title('Convergence Analysis: Accuracy Comparison', fontsize=14)
    ax1.set_ylim(50, 100)
    ax1.set_xticks(df_base['round'])
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=12, loc='lower right')
    
    # Subplot 2: Cumulative Communication Bandwidth
    # Creating arrays for cumulative Megabytes (MB) transmitted over rounds 
    # Formula: (MB per Round * Total Clients) * Round_Number
    round_numbers = df_base['round'].values
    clients_per_round = 2
    
    if orig_kb > 0:
        base_mb_per_round = (orig_kb * clients_per_round) / 1024
        comp_mb_per_round = (comp_kb * clients_per_round) / 1024
        
        cumulative_base_mb = [base_mb_per_round * r for r in round_numbers]
        cumulative_comp_mb = [comp_mb_per_round * r for r in round_numbers]
        
        ax2.plot(round_numbers, cumulative_base_mb, marker='o', linewidth=2.5,
                 markersize=8, color='crimson', alpha=0.9, label='Baseline Bandwidth')
                 
        ax2.plot(round_numbers, cumulative_comp_mb, marker='s', linewidth=2.5,
                 markersize=8, color='forestgreen', alpha=0.9, label='Compressed Bandwidth')
                 
        ax2.fill_between(round_numbers, cumulative_base_mb, cumulative_comp_mb, color='gray', alpha=0.1)
                 
        ax2.set_xlabel('FL Round', fontsize=13)
        ax2.set_ylabel('Cumulative Upload Traffic (Megabytes)', fontsize=13)
        ax2.set_title(f'Bandwidth Overhead (Payload Size Reduction: {reduction:.1f}%)', fontsize=14)
        ax2.set_xticks(round_numbers)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(fontsize=12, loc='upper left')
    else:
        ax2.text(0.5, 0.5, 'Bandwidth Data Error:\nCould not parse Client Log', 
                 horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig('results/comparative_analysis_plot.png', dpi=150)
    print("\nPlot successfully generated: results/comparative_analysis_plot.png")

if __name__ == "__main__":
    main()
