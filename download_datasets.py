from datasets import get_dataset_config_names, load_dataset
from multiprocessing import Pool
import os

def download_branch(branch):
    try:
        print(f"[{os.getpid()}] Downloading {branch}...")
        load_dataset("deepmind/math_dataset", branch, split="test", trust_remote_code=True)
        print(f"[{os.getpid()}] Finished {branch}")
    except Exception as e:
        print(f"[{os.getpid()}] Failed {branch}: {e}")

if __name__ == "__main__":
    branches = get_dataset_config_names("deepmind/math_dataset", trust_remote_code=True)

    print(f"Downloading {len(branches)} branches using process pool...")
    with Pool(processes=8) as pool:  # Change 8 based on available cores
        pool.map(download_branch, branches)
