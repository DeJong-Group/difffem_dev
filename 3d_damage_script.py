import subprocess
import time
import os
import itertools
import multiprocessing
import numpy as np

# Define the combinations


# The target script name
# target_script = "cs_true.py"
target_script_damaged = "3d_damage.py"

max_concurrent_processes = 1


def run_until_done(semaphore, dummy):
    result_filename = f"results/3d_damage_20k.json"
    
    with semaphore:
        while not os.path.exists(result_filename):
            print(f"Result not found. Running script...")
            try:
                subprocess.run(["python", target_script_damaged], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running script: {e}")

        print(f"[{result_filename}] Result file found. Done.")

if __name__ == "__main__":
    semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    # Launch a separate process for each combination
    processes = []
    dummy = None
    p = multiprocessing.Process(target=run_until_done, args=(semaphore, dummy))
    p.start()
    processes.append(p)

    # Optional: Wait for all processes to finish
    for p in processes:
        p.join()
