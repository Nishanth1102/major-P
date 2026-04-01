import os
import subprocess
import time
import shutil
import glob
import sys

# Constants
NUM_CLIENTS = 2
SERVER_SCRIPT = "server.py"
CLIENT_SCRIPT = "client.py"

def run_experiment(use_compression: bool, run_name: str):
    print(f"\n{'='*60}")
    print(f" Starting Experiment: {run_name} (USE_COMPRESSION={use_compression})")
    print(f"{'='*60}")
    
    # Prepare environment variables
    env = os.environ.copy()
    env["USE_COMPRESSION"] = str(use_compression)
    env["TOP_K_RATIO"] = "0.10"
    
    start_time = time.time()
    os.makedirs("results", exist_ok=True)
    
    # 1. Start server
    print(f"[{time.strftime('%H:%M:%S')}] Starting Server...")
    server_process = subprocess.Popen(
        [sys.executable, SERVER_SCRIPT], 
        env=env, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True
    )
    
    time.sleep(3) # Give server time to bind port
    
    # 2. Start clients
    client_processes = []
    for i in range(NUM_CLIENTS):
        print(f"[{time.strftime('%H:%M:%S')}] Starting Client {i}...")
        p = subprocess.Popen(
            [sys.executable, CLIENT_SCRIPT, "--client_id", str(i), "--num_clients", str(NUM_CLIENTS)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        client_processes.append((i, p))
        
    # 3. Read Server output dynamically so we can monitor progress
    server_log_file = f"results/server_log_{run_name}.txt"
    with open(server_log_file, "w") as s_f:
        # iter(server_process.stdout.readline, "") works for streaming
        for line in iter(server_process.stdout.readline, ""):
            if line:
                print(f"[Server] {line.strip()}")
                s_f.write(line)
            if server_process.poll() is not None:
                break
                
    server_process.wait()
    
    # 4. Wait for clients and dump their stdout
    print(f"[{time.strftime('%H:%M:%S')}] Server finished. Gathering client logs...")
    for i, p in client_processes:
        p.wait()
        with open(f"results/client{i}_log_{run_name}.txt", "w") as c_f:
            c_f.write(p.stdout.read())
            
    total_time = time.time() - start_time
    
    # 5. Backup the generated CSV
    if os.path.exists("results/round_metrics.csv"):
        shutil.copy("results/round_metrics.csv", f"results/{run_name}_metrics.csv")
        print(f"[{run_name}] Saved metrics to {run_name}_metrics.csv")
    else:
        print(f"[{run_name}] WARNING: results/round_metrics.csv not found!")
        
    print(f"\n=> {run_name} completed in {total_time:.2f} seconds.\n")
    return total_time

if __name__ == "__main__":
    if not os.path.exists("data/raw") or not glob.glob("data/raw/*.csv"):
        print("Error: No CSV files found in data/raw/. Please add CIC-IDS 2018 datasets.")
        sys.exit(1)
        
    print("NOTE: This will run 2 consecutive FL simulations.")
    print("If you want this to run faster, you can reduce NUM_ROUNDS in server.py and LOCAL_EPOCHS in client.py.")
    
    # Remove old tracking csv if it exists
    if os.path.exists("results/round_metrics.csv"):
        os.remove("results/round_metrics.csv")

    # Run 1: Baseline (Uncompressed)
    time_base = run_experiment(use_compression=False, run_name="baseline")
    
    # Remove tracking csv before start
    if os.path.exists("results/round_metrics.csv"):
        os.remove("results/round_metrics.csv")
        
    # Run 2: Proposed (Compressed)
    time_comp = run_experiment(use_compression=True, run_name="compressed")
    
    # Summary Output
    print("\n" + "#"*60)
    print(" EXPERIMENT BATCH SUMMARY ")
    print("#" * 60)
    print(f"Baseline (Standard FedAvg) Time : {time_base:8.2f} seconds")
    print(f"Compressed (Top-K + FP16) Time  : {time_comp:8.2f} seconds")
    print("\nNext Step:")
    print("Run `python3 generate_analysis.py` to extract Bandwidth parameters and plot Accuracy!")
    print("#"*60 + "\n")
