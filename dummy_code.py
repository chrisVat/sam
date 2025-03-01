import torch
import time
import signal
import sys
import os
import threading

torch.cuda.set_device(3)

ESTABLISH_KILLSWITCH = True

if ESTABLISH_KILLSWITCH:
    from killswitch import setup_killswitch
    setup_killswitch()


def main():
    print("Hello, world!")

    # Create a random tensor with shape (1000, 1000, 5)
    x = torch.randn(1000, 1000, 5).cuda()  # Use GPU 3
    print(x)

    # Simulate long-running computation
    for i in range(1000):
        print(i)
        time.sleep(1)

if __name__ == "__main__":
    main()
