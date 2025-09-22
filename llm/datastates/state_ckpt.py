import torch
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import time
from collections import OrderedDict, deque
import sys
import os
from typing import Union
import pickle
import ctypes
import numpy as np
from datastates.datastates_core import *
from .helper import parse_config, get_checkpoint_version, HOST_CACHE_SIZE, CKPT_PARSER_THREADS
from .utils import get_logger
import atexit
import fasteners
import json

SIZE_UINT64 = ctypes.sizeof(ctypes.c_uint64)
KEY_SEPARATOR = "|"
ALIGNMENT = 4096

class CheckpointEngine:
    def __init__(self, runtime_config={}, rank=0) -> None:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("[DataStates.llm] CUDA is not available. Make sure CUDA drivers are installed and GPU is accessible.")
            
            self.rank           = int(rank)
            datastates_config   = parse_config(runtime_config)
            host_cache_size     = int(datastates_config[HOST_CACHE_SIZE]*(1<<30))       # From GB to Bytes
            cuda_device         = int(torch.cuda.current_device())
            concurrent_parser_threads = int(datastates_config[CKPT_PARSER_THREADS])
            use_uring = True
            self.ckpt_engine    = create_io_engine(host_cache_size, cuda_device, self.rank, use_uring)
            self.executor = ThreadPoolExecutor(max_workers=concurrent_parser_threads)
            self.executor_futures = []
            self.sm = {}
            self.logger = get_logger(__name__)
            self.last_ckpt_version = -1
            self.profile_logs = {}
            # When sigkill is used to stop the process after successful training,
            # the shutdown will not be called, so we register it to atexit
            self._shut = False
            # atexit.register(self.shutdown) 

        except Exception as exc:
            print(f"[DataStates.llm][ERROR] Got exception during DataStates init {exc}")
            sys.exit(-1)

    def save_background(self, state_dict: Union[dict, OrderedDict], path: str):
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
                            "data_offsets": [_start_tensor_offset, _end_tensor_offset],
                        }
                        data = data.contiguous()
                        async_copies[key] = {
                            "tensor": data
                        }
                        _start_tensor_offset = _end_tensor_offset
                        self.sm[version][path].add_var(data, key)
                        snapshot = f"TENSOR{KEY_SEPARATOR}{key}"
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
            profile_log["path"] = path
            profile_log["version"] = version
            profile_log["size"] = _end_tensor_offset
            profile_log["num_tensors"] = len(async_copies)
            if version not in self.profile_logs:
                self.profile_logs[version] = {}
            self.profile_logs[version][path] = profile_log
            return True
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] From DataStates save_background, generated exception: {exc}")
            sys.exit(-1)

    def save(self, state_dict, path: str):
        try:
            if not isinstance(state_dict, (dict, OrderedDict)):
                raise Exception(f"[DataStates.llm] state_dict given to checkpoint must be dictionary. Passed {type(state_dict)} instead for {path}.")
            # future = self.executor.submit(self.save_background, state_dict, path)
            # self.executor_futures.append(future)
            self.save_background(state_dict, path)
            return True
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] Could not save {path}, exception: {exc}, data: {state_dict}")
            sys.exit(-1)
            
    def load(self, path: str, map_location=None):
        try:
            version = get_checkpoint_version(path, self.last_ckpt_version)
            f = open(path, 'rb')
            f.seek(0)
            try:
                header_size_bytes = f.read(SIZE_UINT64)
                header_size = int.from_bytes(header_size_bytes, 'little')
                metadata_size = header_size + SIZE_UINT64
                header = json.loads(f.read(header_size))
            except Exception as exc:
                raise Exception(f"[DataStates.llm] Could not read header size from {path}, exception: {exc}")
            
            [start_offset, end_offset] = np.add(header["datastates_metadata"]["data_offsets"], metadata_size)
            del(header["datastates_metadata"])
            f.seek(start_offset)
            data = pickle.loads(f.read(end_offset-start_offset))

            try:
                restore_list = []
                for k, v in header.items():
                    split_k = deque(k.split(KEY_SEPARATOR))
                    dtype = v["dtype"]
                    if dtype.startswith("torch"):
                        dtype = dtype.replace('torch.', '')
                    shape = v["shape"]
                    [start_offset, end_offset] = np.add(v["data_offsets"], metadata_size)

                    pre_dest = data
                    dest = data
                    while len(split_k):
                        sub_k = split_k.popleft()
                        if sub_k.isdigit():
                            sub_k = int(sub_k) 
                        pre_dest = dest
                        dest = dest[sub_k]
                    if dest != f"TENSOR{KEY_SEPARATOR}{k}":
                        raise Exception(f"[DataStates.llm] The key in header {k} does not match key at location {dest}")

                    # tensor_restored = torch.zeros(size=tuple(shape), dtype=getattr(torch, dtype))
                    # restore_list.append((version, tensor_restored, start_offset, path))
                    f.seek(start_offset)
                    buffer_size = end_offset - start_offset
                    buffer = bytearray(buffer_size)  # Preallocate a writable buffer
                    f.readinto(buffer)  # Read directly into the preallocated buffer
                    tensor_restored = torch.frombuffer(buffer, dtype=getattr(torch, dtype)).reshape(tuple(shape))
                    pre_dest[sub_k] = tensor_restored
                # self.ckpt_engine.load(restore_list)
            except Exception as exc:
                raise Exception(f"[DataStates.llm] Got error with tensor loading {dtype}, {shape}, {exc}")
            # self.logger.info(f"[DataStates.llm] Loaded checkpoint from {path}.")
            return data
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] Could not load {path}, exception: {exc}")
            sys.exit(-1)

    def commit(self, tag):
        # self.wait()
        done, not_done = wait(self.executor_futures, return_when=ALL_COMPLETED)
        assert not not_done, f"Some futures did not complete: {not_done}"
        assert all(future.result() for future in done), "Some futures failed"
        self.executor_futures = []
        queue_stats = self.ckpt_engine.get_queue_stats()
        self.logger.info(f"[DataStates.llm] Checkpoint {tag} on rank {self.rank}: {queue_stats} is ready now!")
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
            queue_stats = self.ckpt_engine.get_queue_stats()
            self.logger.info(f"[DataStates.llm] Wait time in checkpointing engine {time.time()-t} for {self.rank}: {queue_stats}")
            self.logger.info(f"<TIMER:wait-persist-{persist},{time.time()-t}>")
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] From wait, generated exception: {exc}")
            sys.exit(-1)
        return 
    
    def shutdown(self):
        try:
            if self._shut:
                return
            self._shut = True
            self.logger.info("[DataStates.llm] Shutting down CheckpointEngine............")
            self.wait(True, for_all=True)
            self.sm.clear()
            
            perf_profile_file = self.ckpt_engine.shutdown()
            async_profiles = {}
            with open(perf_profile_file, "r", encoding="utf-8", errors="replace") as f:
                async_profiles = json.load(f)
            for path, v in async_profiles.items():
                if path == "version_profiles":
                    # This is a special key for performance profiles of the host flushing for async libraries (e.g., io_uring)
                    self.profile_logs["version_profiles"] = v
                    continue
                version = get_checkpoint_version(path)
                self.profile_logs[version][path].update(v)
                del self.profile_logs[version][path]['path']
                del self.profile_logs[version][path]['version']

            perf_out = {self.rank: self.profile_logs}
            rw_lock = fasteners.InterProcessReaderWriterLock('/dev/shm/state_ckpt.lock')  
            with rw_lock.write_lock():
                print("<"*50)
                print(perf_out)
                print(">"*50)
            self.executor.shutdown(True)
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] From shutdown, generated exception: {exc}")

    def __del__(self):
        try:
            self.shutdown()
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] Got exception during DataStates destructor {exc}")