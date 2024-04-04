import torch
import datastates
import numpy as np
import time

def test_datastates():
    config = {
        "host_cache_size": 1
    }
    print(f"Going to initalize datastates engine...")
    ckpt_engine = datastates.DataStates(config_params=config, rank=0)
    device = torch.device("cpu")    
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        device = torch.device("cuda:0")
    
    tensor_shape = torch.Size([256, 256])
    tensor_dtype = torch.float32
    tensor = torch.randn(tensor_shape, dtype=tensor_dtype).to(device)
    # tensor.uniform_()
    model_name = "datastates_test_model"
    np_array = np.random.randn(512).astype(np.float32)
    ckpt_path = "/dev/shm/datastates-ckpt.pt"
    
    ckpt_obj = {
        "tensor1": tensor,
        "model_name": model_name,
        "rng_iterator": 12345,
        "dtype": tensor_dtype,
        "shape": tensor_shape,
        "random_np_obj": np_array
    }
    print(f"Engine initalized.. Going to checkpoint now...")
    ckpt_engine.save(state_dict=ckpt_obj, path=ckpt_path)
    tensor_sum = torch.sum(tensor)
    ckpt_engine.wait()
    time.sleep(5) # sleep to ensure ckpt file is written

    recovered_obj = ckpt_engine.load(path=ckpt_path)
    recovered_tensor_sum = torch.sum(recovered_obj["tensor1"])
    print(f"Ckpt tensor sum: {tensor_sum}, Recovered tensor sum: {recovered_tensor_sum}")
    # print(recovered_obj)
    del ckpt_engine
    
if __name__ == "__main__":
    test_datastates()


