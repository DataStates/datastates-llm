import torch
import time
from collections import OrderedDict
import sys
from typing import Union
import fasteners
import json
from datastates.engines.helper import parse_config, HOST_CACHE_SIZE, ALIGNMENT, PROFILE_ENGINE
from datastates.utils import get_logger

class BaseCheckpointEngine(object):
    def __init__(self, runtime_config={}, rank=0) -> None:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("[DataStates.llm] CUDA is not available. Make sure CUDA drivers are installed and GPU is accessible.")
            
            self.rank           = int(rank)
            datastates_config   = parse_config(runtime_config)
            self.host_cache_size= int(datastates_config.get(HOST_CACHE_SIZE, 0) * (1 << 30))       # From GB to Bytes
            self.cuda_device    = int(torch.cuda.current_device())
            self.use_uring      = False
            self.profile_engine = bool(datastates_config.get(PROFILE_ENGINE, False))
            self.profile_logs   = {}
            self.logger         = get_logger(__name__)
            self.last_ckpt_version = -1
            self.shut_          = False
            self.ckpt_engine    = None

        except Exception as exc:
            print(f"[DataStates.llm][ERROR] Got exception during DataStates init {exc}")
            sys.exit(-1)

    def get_aligned_offset(self, offset: int) -> int:
        return (offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    def save_(self, state_dict: Union[dict, OrderedDict], path: str):
        pass

    def save(self, state_dict, path: str):
        try:
            if not isinstance(state_dict, (dict, OrderedDict)):
                raise Exception(f"[DataStates.llm] state_dict given to checkpoint must be dictionary. Passed {type(state_dict)} instead for {path}.")
            self.save_(state_dict, path)
            return True
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] Could not save {path}, exception: {exc}, data: {state_dict}")
            sys.exit(-1)
            
    def load(self, path: str, map_location=None):
        pass

    def commit(self, tag):
        pass

    def wait(self, persist=False):
        pass
    
    def shutdown(self):
        if self.shut_:
            return
        self.wait(persist=True)
        self.shut_ = True
        perf_profile_file = self.ckpt_engine.shutdown()
        if not self.profile_engine:
            return
        async_profiles = {}
        with open(perf_profile_file, "r", encoding="utf-8", errors="replace") as f:
            async_profiles = json.load(f)
        for v, info in async_profiles.items():
            self.profile_logs[int(v)].update(info)
            
        perf_out = {self.rank: self.profile_logs}
        rw_lock = fasteners.InterProcessReaderWriterLock('/dev/shm/state_ckpt.lock')  
        with rw_lock.write_lock():
            print("<"*50)
            print(perf_out)
            print(">"*50)
