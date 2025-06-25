import torch
from datastates_core import dstates_engine
import time

def test_ckpt_engine():
    print(f"Going to initalize datastates engine...")
    host_buffer_size = 2 << 30  # 2 GB
    gpu_id = 0
    rank = -1
    ckpt_engine = dstates_engine(host_buffer_size, gpu_id, rank)
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
    ckpt_engine.ckpt(version, 0, tensor1, tensor_bytes, file_offset, ckpt_path)
    file_offset += tensor_bytes
    ckpt_engine.ckpt(version, 1, tensor2, tensor_bytes, file_offset, ckpt_path)
    print(f"Async checkpoint operation launched")
    ckpt_engine.wait(True)

    print(f"Invoking load...")
    # void restore(int version, const char* ptr, const std::uint64_t size, const std::uint64_t file_offset, std::string path);
    rec_tensor1 = tensor1.clone().zero_().cpu()
    rec_tensor2 = tensor2.clone().zero_().cpu()
    file_offset = 0
    ckpt_engine.restore(version, 0, rec_tensor1, tensor_bytes, file_offset, ckpt_path)
    file_offset += tensor_bytes
    ckpt_engine.restore(version, 1, rec_tensor2, tensor_bytes, file_offset, ckpt_path)
    print(f"Loaded checkpoint successfully")
    print("Original tensor1 sum", torch.sum(tensor1), "Recovered tensor1 sum: ", torch.sum(rec_tensor1))
    print("Original tensor2 sum", torch.sum(tensor2), "Recovered tensor2 sum: ", torch.sum(rec_tensor2))

if __name__ == "__main__":
    test_ckpt_engine()


