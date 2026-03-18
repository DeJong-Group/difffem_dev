import subprocess
import time
import os
import itertools
import multiprocessing
import numpy as np

# Define the combinations
specimens = [str(i) for i in np.arange(6)+1]
loadsteps = [str(i) for i in np.arange(6)+1]
freeze_rebars = ['0', '1']
opt_slices = ['0', '1']

lrs = [str(i) for i in [1e8, 1e7, 1e6]]
steps = [str(i) for i in [1000, 500, 250]]

# specimens = ['1']
# loadsteps = ['1', '2', '6']
# loadsteps = ['1']
# freeze_rebars = ['0']
# opt_slices = ['0']

# The target script name
target_script_damaged = "3d_prism_nonzero.py"
max_concurrent_processes = 1

def run_if_not_found(semaphore, specimen, loadstep, freeze_rebar, opt_slice, lrs, steps):
    exp_name = f"3d_prism_nonzero_s{specimen}_ls{loadstep}_freeze_{bool(int(freeze_rebar))}_slice_{bool(int(opt_slice))}"
    result_filename = f"figures/3d_prism_nonzero/{exp_name}_slices.png"
    
    with semaphore:
        for lr in lrs:
            for step in steps:
                filenames = os.listdir('figures/3d_prism_nonzero/')
                filenames_searched = [s.startswith(exp_name) for s in filenames]
                match_sum = sum(filenames_searched)
                if match_sum == 0:
                    print(f"{exp_name} not found, running {lr, step}")
                    try:
                        subprocess.run([
                            "python", 
                            target_script_damaged, 
                            specimen, 
                            loadstep, 
                            freeze_rebar, 
                            opt_slice, 
                            lr, 
                            step], check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error running script: {e}")
                else:
                    print(f"{exp_name} found, skipping")
                    return
        # print(f"[{result_filename}] Result file found. Done.")

def run_once(semaphore, specimen, loadstep, freeze_rebar, opt_slice, lrs, steps):
    exp_name = f"3d_prism_s{specimen}_ls{loadstep}_freeze_{bool(int(freeze_rebar))}_slice_{bool(int(opt_slice))}"
    result_filename = f"figures/3d_prism_nonzero/{exp_name}_slices.png"
    
    with semaphore:
        for lr in lrs:
            for step in steps:
                # print(f"{exp_name} not found, running {lr, step}")
                try:
                    subprocess.run([
                        "python", 
                        target_script_damaged, 
                        specimen, 
                        loadstep, 
                        freeze_rebar, 
                        opt_slice, 
                        lr, 
                        step], check=True)
                    return
                except subprocess.CalledProcessError as e:
                    print(f"Error running script: {e}")

if __name__ == "__main__":
    combinations = list(itertools.product(specimens, loadsteps, freeze_rebars, opt_slices))
    semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    # Launch a separate process for each combination
    processes = []
    for specimen, loadstep, freeze_rebar, opt_slice in combinations:
        print(specimen, loadstep, freeze_rebar, opt_slice)
        p = multiprocessing.Process(target=run_if_not_found, args=(semaphore, specimen, loadstep, freeze_rebar, opt_slice, lrs, steps))
        # p = multiprocessing.Process(target=run_once, args=(semaphore, specimen, loadstep, freeze_rebar, opt_slice, lrs, steps))
        p.start()
        processes.append(p)


    # Optional: Wait for all processes to finish
    for p in processes:
        p.join()
