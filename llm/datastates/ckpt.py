import torch
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import time
from collections import OrderedDict, deque
import sys
from typing import Union
import pickle
import json
import numpy as np
from datastates.datastates_core import create_core_engine
from .helper import parse_config, get_checkpoint_version, HOST_CACHE_SIZE, CKPT_PARSER_THREADS, SIZE_UINT64, KEY_SEPARATOR, ALIGNMENT
from .utils import get_logger

class BaseCheckpointEngine:
    def __init__(self, runtime_config={}, rank=0) -> None:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("[DataStates.llm] CUDA is not available. Make sure CUDA drivers are installed and GPU is accessible.")
            
            self.rank           = int(rank)
            datastates_config   = parse_config(runtime_config)
            host_cache_size     = int(datastates_config[HOST_CACHE_SIZE]*(1<<30))       # From GB to Bytes
            cuda_device         = int(torch.cuda.current_device())
            concurrent_parser_threads = int(datastates_config[CKPT_PARSER_THREADS])
            use_uring = False
            self.ckpt_engine = create_core_engine(host_cache_size, cuda_device, self.rank, use_uring)
            self.executor = ThreadPoolExecutor(max_workers=concurrent_parser_threads)
            self.executor_futures = []

            self.logger = get_logger(__name__)
            self.last_ckpt_version = -1

        except Exception as exc:
            print(f"[DataStates.llm][ERROR] Got exception during DataStates init {exc}")
            sys.exit(-1)

    def get_aligned_offset(self, offset: int) -> int:
        return (offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    def save_background(self, state_dict: Union[dict, OrderedDict], path: str):
        try:
            version = get_checkpoint_version(path, self.last_ckpt_version)
            header = {}
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
                            "tensor": data,
                            "file_offset": _start_tensor_offset
                        }
                        _start_tensor_offset = self.get_aligned_offset(_end_tensor_offset)
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

            lean_state_dict = _parse_state("", state_dict)
            lean_state_dict = pickle.dumps(lean_state_dict, protocol=pickle.HIGHEST_PROTOCOL)
            _end_tensor_offset += self.get_aligned_offset(len(lean_state_dict))
            header.update({"datastates_metadata": {"data_offsets": [_start_tensor_offset, _end_tensor_offset]}})
            header = json.dumps(header).encode("utf-8")
            header_size = len(header).to_bytes(SIZE_UINT64, 'little')   # Force the header size to take 8 bytes
            metadata_size = self.get_aligned_offset(len(header_size) + len(header))
            
            # Launch Async copies
            for i, (_, v) in enumerate(async_copies.items()):
                v["file_offset"] += metadata_size
                tensor_bytes = v["tensor"].numel()*v["tensor"].element_size()
                # print("Checkpointing now region ", i, " of size ", tensor_bytes, " on path ", path)
                self.ckpt_engine.ckpt(version, i, v["tensor"], tensor_bytes, v["file_offset"], path)

            with open(path, 'wb') as f:
                f.seek(0)
                f.write(header_size)
                f.seek(self.get_aligned_offset(len(header_size)))
                f.write(header)
                # Write the lean state dict towards the end of the file.
                f.seek(_start_tensor_offset+metadata_size)
                f.write(lean_state_dict)           
            
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
            header_size_bytes = f.read(SIZE_UINT64)
            header_size = int.from_bytes(header_size_bytes, 'little')
            f.seek(self.get_aligned_offset(SIZE_UINT64))
            header = json.loads(f.read(header_size))
            metadata_size = self.get_aligned_offset(SIZE_UINT64 + header_size)
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
                    # The offsets stored in the header are relative to the start of the data section.
                    # We add the total metadata_size to get the absolute file offsets.
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

                    f.seek(start_offset)
                    buffer_size = end_offset - start_offset
                    buffer = bytearray(buffer_size)
                    f.readinto(buffer)
                    tensor_restored = torch.frombuffer(buffer, dtype=getattr(torch, dtype)).reshape(tuple(shape))
                    pre_dest[sub_k] = tensor_restored
            except Exception as exc:
                raise Exception(f"[DataStates.llm] Got error with tensor loading {dtype}, {shape}, {exc}")

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
        self.last_ckpt_version += 1
        return True

    def wait(self, persist=False):
        try:
            t = time.time()
            self.ckpt_engine.wait(persist)
            queue_stats = self.ckpt_engine.get_queue_stats()
            self.logger.info(f"[DataStates.llm] Wait time in checkpointing engine {time.time()-t} for {self.rank}: {queue_stats}")
            self.logger.info(f"<TIMER:wait-persist-{persist},{time.time()-t}>")
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] From wait, generated exception: {exc}")
            sys.exit(-1)
        return 
    
    def __del__(self):
        self.wait(True)
        self.executor.shutdown(True)
        self.ckpt_engine.shutdown()
        