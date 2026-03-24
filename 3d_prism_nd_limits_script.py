import subprocess
import time
import os
import itertools
import multiprocessing
import numpy as np

# # Define the combinations
specimens = [str(i) for i in np.arange(3)+1]
loadsteps = [str(i) for i in np.arange(1)+1]
freeze_rebars = ['0', '1']
opt_slices = ['0', '1']
weightss = [
    ['0.0', '1.0', '0.0'],
    ['0.0', '0.0', '1.0'],
    ['1.0', '0.0', '0.0'],
    ['0.0', '1.0', '1.0'],
    ['1.0', '1.0', '1.0'],
    ['4500.0', '1.0', '1.0']
]

lrs = [str(i) for i in [1e8, 1e7, 1e6]]
steps = [str(i) for i in [2000, 500, 250]]

# steps = ['200']

# specimens = ['1']
# loadsteps = ['1']
# freeze_rebars = ['1']
# opt_slices = ['0']
weightss = [['0.0', '1.0', '0.0']]

# The target script name
target_script_damaged = "3d_prism_nd_limits.py"
max_concurrent_processes = 3

def run_if_not_found(semaphore, specimen, loadstep, freeze_rebar, opt_slice, lrs, weights, steps):
    exp_name = f"3d_prism_nd_limits_s{specimen}_ls{loadstep}_weights_{weights}_freeze_{bool(int(freeze_rebar))}_slice_{bool(int(opt_slice))}"
    w_rebar = weights[0]
    w_A = weights[1]
    w_B = weights[2]
    with semaphore:
        for lr in lrs:
            for step in steps:
                filenames = os.listdir('figures/3d_prism_nd_limits/')
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
                            w_rebar,
                            w_A,
                            w_B, 
                            step], check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error running script: {e}")
                else:
                    print(f"{exp_name} found, skipping")
                    return
        # print(f"[{result_filename}] Result file found. Done.")

# def run_once(semaphore, specimen, loadstep, freeze_rebar, opt_slice, lrs, steps):
#     exp_name = f"3d_prism_s{specimen}_ls{loadstep}_freeze_{bool(int(freeze_rebar))}_slice_{bool(int(opt_slice))}"
    
#     with semaphore:
#         for lr in lrs:
#             for step in steps:
#                 # print(f"{exp_name} not found, running {lr, step}")
#                 try:
#                     subprocess.run([
#                         "python", 
#                         target_script_damaged, 
#                         specimen, 
#                         loadstep, 
#                         freeze_rebar, 
#                         opt_slice, 
#                         lr, 
#                         step], check=True)
#                     return
#                 except subprocess.CalledProcessError as e:
#                     print(f"Error running script: {e}")

if __name__ == "__main__":
    combinations = list(itertools.product(specimens, loadsteps, freeze_rebars, opt_slices, weightss))
    semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    # Launch a separate process for each combination
    processes = []
    for specimen, loadstep, freeze_rebar, opt_slice, weights in combinations:
        p = multiprocessing.Process(target=run_if_not_found, args=(semaphore, specimen, loadstep, freeze_rebar, opt_slice, lrs, weights, steps))
        # p = multiprocessing.Process(target=run_once, args=(semaphore, specimen, loadstep, freeze_rebar, opt_slice, lrs, steps))
        p.start()
        processes.append(p)


    # Optional: Wait for all processes to finish
    for p in processes:
        p.join()
