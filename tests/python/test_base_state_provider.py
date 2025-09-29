import ctypes
import torch
import numpy as np
import pickle
import json
from datastates.datastates_core import state_manager, create_state_io_engine

# Custom python object to test serialization
class foo():
    def __init__(self, x=-1):
        self.state = x
        self.state_list = [x*i for i in range(1000)]
    def get_state(self):
        return self.state
    def get_state_list(self):
        return self.state_list

def load_to_tensor(base_addr, data_bytes, dtype, shape):
    base_addr = int(base_addr)
    dtype_str = dtype.replace("torch.", "")
    np_dtype = np.dtype(dtype_str)
    tensor_data = np.frombuffer(
        (ctypes.c_char * data_bytes).from_address(base_addr),
        dtype=np_dtype
    )
    shape = tuple(int(dim) for dim in shape.strip("torch.Size").strip("()").strip("[]").split(",") if dim)
    return torch.from_numpy(tensor_data).view(shape)

def test_state_provider():
    a = torch.randn(100, 100, device='cuda')        # GPU tensor
    b = a.to("cpu").pin_memory()                    # Pinned host tensor
    c = a.to("cpu").detach().clone()                # Unpinned host tensor
    d = {"key1": "value1", "key2": "value2", "key3": "value3"}  # A dictionary
    d_serialized = pickle.dumps(d)
    e = foo(-3)
    e_serialized = pickle.dumps(e)
    sm = state_manager()
    sm.add_var(a, "tensor_a")
    sm.add_var(b, "tensor_b")
    sm.add_var(c, "tensor_c")
    sm.add_var(d_serialized, "dict_d")
    sm.add_var(e_serialized, "foo_e")
    sm.print_state()

    ckpt_engine = create_state_io_engine(host_cache_size=1<<30, gpu_id=0, rank=0, use_io_uring=True) # 1GB host cache
    path = "/tmp/test_state_0.dstates_ckpt"
    ckpt_engine.ckpt(0, sm, path)
    ckpt_engine.wait(sm, persist=True)
    print(f"Checkpointing version 0 now completed.")

    ckpt_engine.wait(sm, persist=True)
    restored_state = ckpt_engine.restore(0, path)
    restored_state = json.loads(restored_state) # Convert JSON string to dict
    for k, v in restored_state.items():
        start_offset, end_offset = v["offsets"]
        size = end_offset - start_offset
        if "tensor_" in k:
            dtype = v['dtype']
            shape = v['shape']
            base_addr = v['ptr']
            tensor = load_to_tensor(base_addr, size, dtype, shape)
            print(f"Restored {k}: dtype={dtype}, shape={shape}, size={size}, base_addr={base_addr}")
            restored_state[k] = tensor
        elif k == "dict_d" or k == "foo_e":
            base_addr = v['ptr']
            dict_data = (ctypes.c_char * size).from_address(int(base_addr))
            dict_serialized = bytes(bytearray(dict_data))
            restored_dict = pickle.loads(dict_serialized)
            print(f"Restored {k}: {restored_dict}")
            restored_state[k] = restored_dict
    assert torch.allclose(a.cpu(), restored_state["tensor_a"].cpu()), "Restored tensor_a is not same as original"
    assert torch.allclose(b.cpu(), restored_state["tensor_b"].cpu()), "Restored tensor_b is not same as original"
    assert torch.allclose(c.cpu(), restored_state["tensor_c"].cpu()), "Restored tensor_c is not same as original"
    assert d == restored_state["dict_d"], "Restored dict_d is not same as original"
    assert e.get_state() == restored_state["foo_e"].get_state(), "Restored foo_e state is not same as original"
    assert e.get_state_list() == restored_state["foo_e"].get_state_list(), "Restored foo_e state_list is not same as original"
    print("All verification assertions passed. State restored successfully and matches original.")
    ckpt_engine.shutdown() # Shutdown the engine to avoid CUDA context being deleted while engine holds GPU resources

if __name__ == "__main__":
    test_state_provider()