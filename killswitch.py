import os
import time
import threading
import torch

KILL_FILE = "/data2/shared/chris_shared_file"

def setup_killswitch():
    # print cuda visible devices
    #print(f"*** CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
    cuda_visible_devices = None

    if torch.cuda.is_available():   
        cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    
    MY_KILL_FILE = KILL_FILE
    if cuda_visible_devices:
        MY_KILL_FILE = f"{KILL_FILE}_{cuda_visible_devices}"

    
    if not os.path.exists(MY_KILL_FILE):
        with open(MY_KILL_FILE, "w") as f:
            f.write("")
    os.chmod(MY_KILL_FILE, 0o777)  # Make sure anyone can modify it
    
    def check_kill_file():
        while True:
            time.sleep(5) 
            if not os.path.exists(MY_KILL_FILE):
                print("Kill file not found. Exiting.")
                os._exit(1)  

    checker_thread = threading.Thread(target=check_kill_file, daemon=True)
    checker_thread.start()
