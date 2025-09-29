import time
from datastates import BaseCheckpointEngine
import torch
import numpy as np

# Custom python object to test serialization
class foo():
    def __init__(self, x=-1):
        self.state = x
        self.state_list = [x*i for i in range(1000)]
    def get_state(self):
        return self.state
    def get_state_list(self):
        return self.state_list

def test_ckpt_base_engine():
    print(f"Going to initalize datastates base engine...")
    config = {
        "host_cache_size": 2,  # 2 GB
        "parser_threads": 1,  # 1 thread (dummy arg for now)
    }
    device = torch.device("cpu")    
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        device = torch.device("cuda:0")
    
    ckpt_engine = BaseCheckpointEngine(config, rank=0)
    
    tensor_shape = torch.Size([256, 256])
    tensor_dtype = torch.bfloat16
    tensor1 = torch.randn(tensor_shape, dtype=tensor_dtype).to(device)
    tensor2 = torch.randn(tensor_shape, dtype=torch.float32).to("cpu")
    custom_py_obj = foo(-5)
    np_array = np.random.randn(512).astype(np.float32)
    state_dict = {
        "tensor1": tensor1,
        "tensor2": tensor2,
        "custom_py_obj": custom_py_obj,
        "random_np_obj": np_array,
        "test_string": "this is a random test string"*100,
    }

    ckpt_path = "/tmp/datastates-ckpt.pt"
    version = 1

    print(f"Invoking async checkpoint...")
    ckpt_engine.save(state_dict, ckpt_path)
    print(f"Async checkpoint operation launched")
    ckpt_engine.wait(True)

    print(f"Invoking load...")
    restored_state = ckpt_engine.load(ckpt_path)
    print(f"Loaded checkpoint successfully")
    assert torch.allclose(tensor1.cpu(), restored_state["tensor1"].cpu()), "Restored tensor1 is not same as original"
    assert torch.allclose(tensor2.cpu(), restored_state["tensor2"].cpu()), "Restored tensor2 is not same as original"
    assert custom_py_obj.get_state() == restored_state["custom_py_obj"].get_state(), "Restored custom_py_obj state is not same as original"
    assert custom_py_obj.get_state_list() == restored_state["custom_py_obj"].get_state_list(), "Restored custom_py_obj state_list is not same as original"
    assert np.array_equal(np_array, restored_state["random_np_obj"]), "Restored random_np_obj is not same as original"
    assert "this is a random test string" in restored_state["test_string"], "Restored test_string is not same as original"
    print(f"Verified restored tensors and objects are same as original")
    del ckpt_engine # Shutdown the engine to avoid CUDA context being deleted while engine holds GPU resources

if __name__ == "__main__":
    test_ckpt_base_engine()