import torch
import datastates_ckpt
import numpy as np
import time

# We need to test the way deepspeed checkpointing config is setup
# The DeepSpeed config, along with other params such as 
# 'zero_optimization_stage', 'pipeline_parallel' 'tensor_parallel' etc.
# From there, we need datastate_config, which has config
# ['host_cache_size'] attribute.

class DeepSpeedConfig:
    def __init__(self):
        self.datastates_config = self.DataStatesConfig()
    
    class DataStatesConfig:
        def __init__(self):
            self.enabled = True
            self.config = {
                "host_cache_size": 1,
                "parser_threads": 2,
                "pin_host_cache": True
            }

def test_ckpt_engine():
    deepspeed_config = DeepSpeedConfig()
    print(f"Going to initalize datastates engine...")
    ckpt_engine = datastates_ckpt.Checkpointing(deepspeed_config=deepspeed_config, rank=0)
    device = torch.device("cpu")    
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        device = torch.device("cuda:0")
    
    tensor_shape = torch.Size([256, 256])
    tensor_dtype = torch.float32
    tensor1 = torch.randn(tensor_shape, dtype=tensor_dtype).to(device)
    tensor2 = tensor1.to(torch.float16)

    ckpt_path = "/dev/shm/datastates-ckpt.pt"

    file_offset = 0
    version = 1
    tensors = [
        (version, tensor1, file_offset, ckpt_path),
        (version, tensor2, file_offset+tensor1.untyped_storage().size(), ckpt_path),
    ]

    print(f"Invoking async checkpoint...")
    ckpt_engine.async_save(tensors)
    ckpt_engine.wait()

    print(f"Invoking load...")
    rec_tensor1 = tensor1.clone().zero_()
    rec_tensor2 = tensor2.clone().zero_()
    rec_tensors = [
        (version, tensor1, file_offset, ckpt_path),
        (version, tensor2, file_offset+rec_tensor1.untyped_storage().size(), ckpt_path),
    ]
    ckpt_engine.load(rec_tensors)
        
if __name__ == "__main__":
    test_ckpt_engine()


