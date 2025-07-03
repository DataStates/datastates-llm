import torch
import time
from datastates.datastates_core import state_manager, create_io_engine

a = torch.randn(100, 100, device='cuda')        # GPU tensor
b = a.to("cpu").pin_memory()                    # Pinned host tensor
c = a.to("cpu").detach().clone()                # Unpinned host tensor
d = {"key1": "value1", "key2": "value2", "key3": "value3"}  # A dictionary
sm = state_manager()
sm.add_var(a)
sm.add_var(b)
sm.add_var(c)
sm.add_var(d)
sm.print_state()

io_engine = create_io_engine(host_cache_size=1<<30, gpu_id=0)
nruns = 5
for i in range(nruns):
    persist = True if i == nruns - 1 else False
    d.update({f"new_key_{i}": f"new_value_{i}"})  # Update the dictionary with new key-value pairs
    print(f"Checkpointing {i}", end=' ')
    io_engine.ckpt(i, sm, f"/dev/shm/test_state_{i}.dstates_ckpt")
    io_engine.wait(sm, persist)
    print(f"... now completed.")

##### Outputs the following #####
# Number of Registered Providers: 4
# Provider region: 1, Tier: GPU_TIER, Serialized Data Size: 40000, Is Tensor: Yes, Is Serialized: Yes
# Provider region: 2, Tier: HOST_PINNED_TIER, Serialized Data Size: 40000, Is Tensor: Yes, Is Serialized: Yes
# Provider region: 3, Tier: HOST_UNPINNED_TIER, Serialized Data Size: 40000, Is Tensor: Yes, Is Serialized: Yes
# Provider region: 4, Tier: HOST_UNPINNED_TIER, Serialized Data Size: 64, Is Tensor: No, Is Serialized: No
# Checkpointing 0 ... now completed.
# Checkpointing 1 ... now completed. ...