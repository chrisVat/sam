
import torch

for i in range(torch.cuda.device_count()):
    allocated = torch.cuda.memory_allocated(i) / 1024**2  # Convert to MiB
    reserved = torch.cuda.memory_reserved(i) / 1024**2    # Convert to MiB
    free_reserved = reserved - allocated

    print(f"GPU {i}:")
    print(f"  Allocated: {allocated:.2f} MiB")
    print(f"  Reserved: {reserved:.2f} MiB")
    print(f"  Free (Reserved - Allocated): {free_reserved:.2f} MiB")
                                
