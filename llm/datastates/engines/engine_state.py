import torch
import time
from collections import OrderedDict
import sys
from typing import Union
import pickle
import ctypes
from datastates.datastates_core import create_state_io_engine, state_manager
from datastates.engines.helper import get_checkpoint_version, KEY_SEPARATOR
from datastates.engines.engine_base import BaseCheckpointEngine
import json

class StateCheckpointEngine(BaseCheckpointEngine):
    def __init__(self, runtime_config={}, rank=0) -> None:
        try:
            super().__init__(runtime_config, rank)
            self.use_uring = True
            self.ckpt_engine    = create_state_io_engine(self.host_cache_size, self.cuda_device, self.rank, self.use_uring)
            self.sm = {}

        except Exception as exc:
            print(f"[DataStates.llm][ERROR] Got exception during DataStates init {exc}")
            sys.exit(-1)

    def save_(self, state_dict: Union[dict, OrderedDict], path: str):
        try:
            profile_log = {}
            version = get_checkpoint_version(path, self.last_ckpt_version)
            self.last_ckpt_version = version
            header = {}
            if version not in self.sm:
                self.sm[version] = {}
            assert path not in self.sm[version], f"[DataStates.llm] Path {path} already exists in state manager for version {version}, having keys {self.sm[version].keys()}"
            self.sm[version][path] = state_manager()
            async_copies = {}
            _start_tensor_offset = 0
            _end_tensor_offset = 0

            def _parse_state(key, data):
                nonlocal _start_tensor_offset, _end_tensor_offset
                try:
                    if torch.is_tensor(data): # and data.device.type == 'cuda':
                        tensor_size = data.numel()*data.element_size()
                        _end_tensor_offset += tensor_size
                        header[key] = {
                            "dtype": str(data.dtype),                       # JSON cannot stringify torch.Size() type
                            "shape": tuple(data.shape),
                            "offsets": [_start_tensor_offset, _end_tensor_offset],
                        }
                        data = data.contiguous()
                        async_copies[key] = {
                            "tensor": data
                        }
                        _start_tensor_offset = _end_tensor_offset
                        mapped_key = f"TENSOR{KEY_SEPARATOR}{key}"
                        self.sm[version][path].add_var(data, mapped_key)
                        snapshot = mapped_key
                    elif isinstance(data, list):
                        snapshot = [None]*len(data)
                        for (idx, ele) in enumerate(data):
                            new_key = f"{key}{KEY_SEPARATOR}{idx}" if len(key) else f"{idx}"
                            snapshot[idx] = _parse_state(new_key, ele)
                    elif isinstance(data, (dict, OrderedDict)):
                        snapshot = {}
                        for (k, v) in data.items():
                            new_key = f"{key}{KEY_SEPARATOR}{k}" if len(key) else f"{k}"
                            snapshot[k] = _parse_state(new_key, v)
                    else:
                        snapshot = data
                    return snapshot
                except Exception as exc:
                    raise Exception(f"[DataStates.llm][ERROR] Cannot parse {key}, exception: {exc}, data is {data}")

            t = time.time()
            lean_state_dict = _parse_state("", state_dict)
            profile_log["parse_time"] = time.time() - t
            t = time.time()
            lean_state_dict = pickle.dumps(lean_state_dict, protocol=pickle.HIGHEST_PROTOCOL)
            profile_log["pickle_time"] = time.time() - t
            _end_tensor_offset += len(lean_state_dict)
            self.sm[version][path].add_var(lean_state_dict, "datastates_metadata")

            t = time.time()
            self.ckpt_engine.ckpt(version, self.sm[version][path], path)
            profile_log["ckpt_time"] = time.time() - t
            profile_log["size"] = _end_tensor_offset
            profile_log["num_tensors"] = len(async_copies)
            if version not in self.profile_logs:
                self.profile_logs[version] = {}
            self.profile_logs[version][path.split("/")[-1]] = profile_log
            return True
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] From DataStates save_background, generated exception: {exc}")
            sys.exit(-1)

    def load(self, path: str, map_location=None):
        try:
            version = get_checkpoint_version(path, self.last_ckpt_version)
            header = self.ckpt_engine.restore(version, path)
            header = json.loads(header)
            lean_state_dict_info = header.get("datastates_metadata", None)
            assert lean_state_dict_info is not None, f"[DataStates.llm] No metadata found in header for {path}"
            buffer_size = lean_state_dict_info["size"]
            buffer_ptr = int(lean_state_dict_info["ptr"])
            lean_buffer = (ctypes.c_char * buffer_size).from_address(buffer_ptr)
            state_dict = pickle.loads(lean_buffer)
            def _reconstruct_state(key, snapshot):
                try:
                    if isinstance(snapshot, str) and snapshot.startswith(f"TENSOR{KEY_SEPARATOR}"):
                        header_info = header.get(snapshot, None)
                        assert header_info is not None, f"Key {key} not found in header ({header.keys()}) for {path}"
                        base_addr = int(header_info["ptr"])
                        dtype_str = header_info["dtype"].replace("torch.", "")
                        start, end = header_info["offsets"]
                        tensor_shape = tuple(int(dim) for dim in header_info["shape"].strip("torch.Size").strip("()").strip("[]").split(",") if dim)
                        tensor_size = end - start
                        c_buffer = (ctypes.c_char * tensor_size).from_address(base_addr)
                        torch_dtype = getattr(torch, dtype_str)
                        element_count = tensor_size // torch_dtype.itemsize
                        snapshot = torch.frombuffer(c_buffer, dtype=torch_dtype, count=element_count).view(tensor_shape)
                        if map_location is not None:
                            if callable(map_location):
                                # Emulate torch.load(map_location) behavior
                                device = map_location(None, snapshot.device)
                                snapshot = snapshot.to(device)
                            else:
                                snapshot = snapshot.to(map_location)
                        return snapshot

                    elif isinstance(snapshot, list):
                        return [
                            _reconstruct_state(f"{key}{KEY_SEPARATOR}{idx}" if key else str(idx), ele)
                            for idx, ele in enumerate(snapshot)
                        ]
                    elif isinstance(snapshot, (dict, OrderedDict)):
                        return {
                            k: _reconstruct_state(f"{key}{KEY_SEPARATOR}{k}" if key else k, v)
                            for k, v in snapshot.items()
                        }
                    return snapshot
                except Exception as exc:
                    raise Exception(f"[DataStates.llm][ERROR] Cannot reconstruct {key}, exception: {exc}, snapshot is {snapshot}")
            return _reconstruct_state("", state_dict)
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] Could not load {path}, exception: {exc}")
            sys.exit(-1)

    def commit(self, tag):
        self.logger.info(f"[DataStates.llm] Checkpoint {tag} on rank {self.rank} is ready now!")
        # self.last_ckpt_version += 1
        return True

    def wait(self, persist=False, for_all=False):
        try:
            if not self.sm:
                self.logger.info("[DataStates.llm] No checkpoints to wait for.")
                return
            t = time.time()
            assert self.last_ckpt_version in self.sm, f"[DataStates.llm] Last checkpoint version {self.last_ckpt_version} not found in state manager."
            sms_to_wait_for = [self.last_ckpt_version]
            if for_all:
                sms_to_wait_for = list(self.sm.keys())  
            for smid in sms_to_wait_for:
                for k, mgr in self.sm[smid].items():
                    self.ckpt_engine.wait(mgr, persist)
            self.profile_logs[self.last_ckpt_version][f"wait_time_persist_{persist}"] = time.time() - t
            self.logger.info(f"[DataStates.llm] <TIMER:wait-persist-{persist},{time.time()-t}>")
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] From wait, generated exception: {exc}")
            sys.exit(-1)
        return 
    
    def shutdown(self):
        try:
            if self.shut_:
                return
            self.wait(True, for_all=True)
            self.sm.clear()
            super().shutdown()
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] From shutdown, generated exception: {exc}")

    def __del__(self):
        try:
            self.shutdown()
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] Got exception during DataStates destructor {exc}")