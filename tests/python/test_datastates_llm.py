import torch
from datastates import CheckpointEngine as CheckpointEngine
import numpy as np
import time
import os

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

def test_datastates():
    deepspeed_config = DeepSpeedConfig()
    print(f"Going to initalize datastates engine...")
    ckpt_engine = CheckpointEngine(runtime_config=deepspeed_config, rank=0)
    device = torch.device("cpu")    
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        device = torch.device("cuda:0")
    
    tensor_shape = torch.Size([25600, 2560])
    tensor_dtype = torch.float32
    tensor = torch.randn(tensor_shape, dtype=tensor_dtype).to(device)
    tensor2 = torch.randn(tensor_shape, dtype=tensor_dtype).to('cpu').pin_memory()
    
    model_name = "datastates_test_model"
    np_array = np.random.randn(512).astype(np.float32)
    ckpt_path = "/tmp/datastates-ckpt.pt"
    
    ckpt_obj = {
        "tensor1": tensor,
        "tensor2": tensor2,
        "model_name": model_name,
        "rng_iterator": 12345,
        "dtype": tensor_dtype,
        "shape": tensor_shape,
        "random_np_obj": np_array,
        "test_string": "this is a random test string"*100,
    }
    
    print(f"Engine initalized.. Going to checkpoint now...")
    start_time = time.time()
    ckpt_engine.save(state_dict=ckpt_obj, path=ckpt_path)
    end_time = time.time()
    print("Checkpointing in time ", end_time-start_time, " tensor of sum: ", torch.sum(ckpt_obj["tensor1"]), torch.sum(ckpt_obj["tensor2"]))
    ckpt_engine.wait(True)
    f = open(ckpt_path, "rb")
    os.fsync(f.fileno())
    f.close()
    wait_time = time.time()
    print(f"Checkpointing completed successfully in {wait_time - start_time}, now recovering the checkpoint...")

    recovered_obj = ckpt_engine.load(path=ckpt_path)
    print("Recovering tensor of sum: ", torch.sum(recovered_obj["tensor1"]), torch.sum(recovered_obj["tensor2"]))
    print(f"Checkpoint recovered successfully (note that sums maybe slightly different due to floating point precision)")
    del ckpt_engine
    
if __name__ == "__main__":
    test_datastates()


