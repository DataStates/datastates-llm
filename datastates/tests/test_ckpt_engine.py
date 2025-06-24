import torch
from datastates_core import handle as CkptEngine
import time

def test_ckpt_engine():
    print(f"Going to initalize datastates engine...")
    ckpt_engine = CkptEngine((2 << 30), 0, 0)
    device = torch.device("cpu")    
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        device = torch.device("cuda:0")
    
    tensor_shape = torch.Size([256, 256])
    tensor_dtype = torch.bfloat16
    tensor1 = torch.randn(tensor_shape, dtype=tensor_dtype).to(device)
    tensor2 = torch.randn(tensor_shape, dtype=tensor_dtype).to("cpu")
    tensor_bytes = tensor1.numel()*tensor1.element_size()

    ckpt_path = "/dev/shm/datastates-ckpt.pt"

    file_offset = 0
    version = 1

    print(f"Invoking async checkpoint...")
    ckpt_engine.ckpt(version, tensor1, tensor_bytes, file_offset, ckpt_path)
    file_offset += tensor_bytes
    ckpt_engine.ckpt(version, tensor2, tensor_bytes, file_offset, ckpt_path)
    print(f"Async checkpoint operation launched")
    ckpt_engine.wait()
    time.sleep(5) # Sleep for 10s to ensure that the file is written before we start reading it back.

    print(f"Invoking load...")
    # void restore(int version, const char* ptr, const std::uint64_t size, const std::uint64_t file_offset, std::string path);
    rec_tensor1 = tensor1.clone().zero_().cpu()
    rec_tensor2 = tensor2.clone().zero_().cpu()
    file_offset = 0
    ckpt_engine.restore(version, rec_tensor1, tensor_bytes, file_offset, ckpt_path)
    file_offset += tensor_bytes
    ckpt_engine.restore(version, rec_tensor2, tensor_bytes, file_offset, ckpt_path)
    print(f"Loaded checkpoint successfully")
    print("Original tensor1 sum", torch.sum(tensor1), "Recovered tensor1 sum: ", torch.sum(rec_tensor1))
    print("Original tensor2 sum", torch.sum(tensor2), "Recovered tensor2 sum: ", torch.sum(rec_tensor2))

if __name__ == "__main__":
    test_ckpt_engine()


