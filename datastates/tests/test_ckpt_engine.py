import torch
from datastates.ckpt import CkptEngine
import time

def test_ckpt_engine():
    print(f"Going to initalize datastates engine...")
    ckpt_engine = CkptEngine(host_cache_size=(2 << 30), gpu_device_id=0, rank=0)
    device = torch.device("cpu")    
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        device = torch.device("cuda:0")
    
    tensor_shape = torch.Size([256, 256])
    tensor_dtype = torch.bfloat16
    tensor1 = torch.randn(tensor_shape, dtype=tensor_dtype).to(device)
    tensor2 = torch.randn(tensor_shape, dtype=tensor_dtype).to(device)
    tensor_bytes = tensor1.numel()*tensor1.element_size()

    ckpt_path = "/dev/shm/datastates-ckpt.pt"

    file_offset = 0
    version = 1
    tensors = [
        (version, tensor1, file_offset, ckpt_path),
        (version, tensor2, file_offset+tensor_bytes, ckpt_path),
    ]

    print(f"Invoking async checkpoint...")
    ckpt_engine.async_save(tensors)
    ckpt_engine.wait()
    time.sleep(5) # Sleep for 10s to ensure that the file is written before we start reading it back.

    print(f"Invoking load...")
    rec_tensor1 = tensor1.clone().zero_().cpu()
    rec_tensor2 = tensor2.clone().zero_().cpu()
    rec_tensors = [
        (version, rec_tensor1, file_offset, ckpt_path),
        (version, rec_tensor2, file_offset+tensor_bytes, ckpt_path),
    ]
    ckpt_engine.load(rec_tensors)
    print(f"Loaded checkpoint successfully")
        
if __name__ == "__main__":
    test_ckpt_engine()


