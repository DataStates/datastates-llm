from datastates.ckpt.src import handle as datastates_handle
import torch
import time
import sys
import os
from datastates.utils import get_logger

# Acts as a Python Interfact to manage the CPP checkpoint engine
class CkptEngine:
    def __init__(self, host_cache_size, gpu_device_id, rank) -> None:
        try:
            self.ckpt_engine = datastates_handle(host_cache_size, gpu_device_id, rank)
            self.logger = get_logger(__name__)
            self.last_ckpt_version = -1
        except Exception as exc:
            print(f"[DataStates.ckpt][ERROR] Got exception during DataStates init {exc}")
            sys.exit(-1)

    # This function accepts a list of tuples containing tensors to checkpoint.
    # Each tuple contains: version, torch.Tensor, file offset, and path
    def async_save(self, tensors: list[tuple[int, torch.Tensor, int, str]]):
        try:
            for t in tensors:
                version, tensor, file_offset, path = t
                tensor_bytes = tensor.numel()*tensor.element_size()
                assert tensor_bytes > 0, "Tensor size should be > 0"
                self.ckpt_engine.ckpt_tensor(version, tensor, tensor_bytes, file_offset, path)
        except Exception as exc:
            self.logger.error(f"[DataStates.ckpt][ERROR][async_save] {exc}")
            sys.exit(-1)

    def load(self, tensors: list[tuple[int, torch.Tensor, int, str]]):
        try:
            for t in tensors:
                version, tensor, file_offset, path = t
                file_size = os.path.getsize(path)
                tensor_bytes = tensor.numel()*tensor.element_size()
                assert tensor_bytes > 0, "Tensor size should be > 0"
                assert file_offset + tensor_bytes <= file_size, f"Tensor at offset {file_offset} overflows file size {file_size}"
                self.ckpt_engine.restore_tensor(version, tensor, tensor_bytes, file_offset, path)
                self.logger.info(f"[Datastates.ckpt] Restored tensor {tensor_bytes} from {file_offset}")
        except Exception as exc:
            self.logger.error(f"[DataStates.ckpt][ERROR][load] {exc}")
            sys.exit(-1)
    
    def commit(self, tag):
        self.wait()
        self.logger.info(f"[DataStates.ckpt] Checkpoint {tag} is ready now!")
        self.last_ckpt_version += 1
        return True

    def wait(self):
        try:
            self.ckpt_engine.wait()
        except Exception as exc:
            self.logger.error(f"[DataStates.ckpt][ERROR] From wait, generated exception: {exc}")
            sys.exit(-1)
    
    def __del__(self):
        return self.ckpt_engine.shutdown()