import time

import torch

from src.blur.full import full_blur
from src.privacy_network.osv5m import out_dir


if __name__ == "__main__":
    x = torch.rand(16, 3, 224, 224, device="cuda") 
    full_blur(x)  
    torch.cuda.synchronize()

    start = time.time()
    for i in range(20):
        full_blur(x)
        
    torch.cuda.synchronize() 
    seconds = (time.time() - start) / (20 * len(x))

    (out_dir / "full_sigma20_seconds.txt").write_text(f"{seconds:.6f}")
    print(f"full blur: {seconds:.6f} seconds per photo")
