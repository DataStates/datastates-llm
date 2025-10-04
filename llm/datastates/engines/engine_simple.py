import torch
import time
from collections import OrderedDict, deque
import sys
from typing import Union
import pickle
import json
import numpy as np
from datastates.datastates_core import create_core_engine
from datastates.engines.helper import get_checkpoint_version, SIZE_UINT64, KEY_SEPARATOR
from datastates.engines.engine_base import BaseCheckpointEngine

class SimpleCheckpointEngine(BaseCheckpointEngine):
    def __init__(self, runtime_config={}, rank=0) -> None:
        try:
            super().__init__(runtime_config, rank)
            self.use_uring           = False # In simple engine we do not use uring
            self.profile_engine      = False # In simple engine, we do not have the profiler.
            self.ckpt_engine = create_core_engine(self.host_cache_size, self.cuda_device, self.rank, self.use_uring)
        except Exception as exc:
            print(f"[DataStates.llm][ERROR] Got exception during DataStates init {exc}")
            sys.exit(-1)

    def save_(self, state_dict: Union[dict, OrderedDict], path: str):
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
                            "offsets": [_start_tensor_offset, _end_tensor_offset],
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
            header.update({"datastates_metadata": {"offsets": [_start_tensor_offset, _end_tensor_offset]}})
            header = json.dumps(header).encode("utf-8")
            header_size = len(header).to_bytes(SIZE_UINT64, 'little')   # Force the header size to take 8 bytes
            metadata_size = self.get_aligned_offset(len(header_size)) + self.get_aligned_offset(len(header))
            
            # Launch Async copies
            for i, (_, v) in enumerate(async_copies.items()):
                v["file_offset"] += metadata_size
                tensor_bytes = v["tensor"].numel()*v["tensor"].element_size()
                # print("Checkpointing now region ", i, " of size ", tensor_bytes, " on path ", path)
                self.ckpt_engine.ckpt(version, i, v["tensor"], tensor_bytes, v["file_offset"], path)

            with open(path, 'wb') as f:
                f.seek(0)
                f.write(header_size)
                f.seek(self.get_aligned_offset(SIZE_UINT64))
                f.write(header)
                # Write the lean state dict towards the end of the file.
                f.seek(_start_tensor_offset+metadata_size)
                f.write(lean_state_dict)           
            
            return True
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] From DataStates save_, generated exception: {exc}")
            sys.exit(-1)
            
    def load(self, path: str, map_location=None):
        try:
            version = get_checkpoint_version(path, self.last_ckpt_version)
            f = open(path, 'rb')
            f.seek(0)
            header_size_bytes = f.read(SIZE_UINT64)
            header_size = int.from_bytes(header_size_bytes, 'little')
            f.seek(self.get_aligned_offset(SIZE_UINT64))
            raw_header = f.read(header_size)
            header = json.loads(raw_header)
            metadata_size = self.get_aligned_offset(SIZE_UINT64) + self.get_aligned_offset(header_size)
            [start_offset, end_offset] = np.add(header["datastates_metadata"]["offsets"], metadata_size)
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
                    [start_offset, end_offset] = np.add(v["offsets"], metadata_size)

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
        self.logger.info(f"[DataStates.llm] Checkpoint {tag} on rank {self.rank} is ready now!")
        # self.last_ckpt_version += 1
        return True

    def wait(self, persist=False):
        try:
            t = time.time()
            self.ckpt_engine.wait(persist)
            self.logger.info(f"[DataStates.llm] <TIMER:wait-persist-{persist},{time.time()-t}>")
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] From wait, generated exception: {exc}")
            sys.exit(-1)
        return 
    
    def shutdown(self):
        try:
            if self.shut_:
                return
            self.wait(True)
            super().shutdown()
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] From shutdown, generated exception: {exc}")
            sys.exit(-1)
        return

    def __del__(self):
        self.shutdown()
        return