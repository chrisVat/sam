import os
import time
import threading


KILL_FILE = "/data2/shared/chris_shared_file"

def setup_killswitch():
    if not os.path.exists(KILL_FILE):
        with open(KILL_FILE, "w") as f:
            f.write("")
    os.chmod(KILL_FILE, 0o777)  # Make sure anyone can modify it
    
    def check_kill_file():
        while True:
            time.sleep(5) 
            if not os.path.exists(KILL_FILE):
                print("Kill file not found. Exiting.")
                os._exit(1)  

    checker_thread = threading.Thread(target=check_kill_file, daemon=True)
    checker_thread.start()
