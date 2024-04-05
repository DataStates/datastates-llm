from datastates_cpp import handle as datastates_handle
import torch
from concurrent.futures import ThreadPoolExecutor
import time
from collections import OrderedDict, deque
import sys
from typing import Union
import pickle
import json
import ctypes
import numpy as np
import logging

SIZE_UINT64 = ctypes.sizeof(ctypes.c_uint64)
KEY_SEPARATOR = "|"

class DataStates:
    def __init__(self, deepspeed_config, rank) -> None:
        try:
            self.rank = rank
            datastates_config = deepspeed_config.datastates_config.config
            self.ckpt_engine = datastates_handle((datastates_config['host_cache_size'] << 30), 
                                            int(torch.cuda.current_device()),
                                            int(self.rank)
                                        )
            self.futures = None
            concurrent_parser_threads = 4
            self.executor = ThreadPoolExecutor(max_workers=concurrent_parser_threads)

            logging_level = logging.INFO
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s")
            channel = logging.StreamHandler(stream=sys.stdout)
            channel.setLevel(logging_level)
            channel.setFormatter(formatter)
            self.logger = logging.getLogger("DataStates")
            self.logger.setLevel(logging_level)
            self.logger.addHandler(channel)
        except Exception as exc:
            print(f"[DataStates][ERROR] Got exception during DataStates init {exc}")
            sys.exit(-1)

    def save_background(self, state_dict: Union[dict, OrderedDict], path: str):
        try:
            if not isinstance(state_dict, dict) and not isinstance(state_dict, OrderedDict):
                raise Exception(f"state_dict given to checkpoint must be dictionary. Passed {type(state_dict)} instead for {path}.")
            try:
                version = int(path.split("/")[-2].replace('global_step', ''))
            except Exception as exc:
                # If not running with deepspeed (which has directory path of format `global_step{version}/filename.pt`)
                # use a static version number.
                version = 0

            header = {}
            async_copies = {}
            _start_tensor_offset = 0
            _end_tensor_offset = 0
            def _parse_state(key, data):
                nonlocal _start_tensor_offset, _end_tensor_offset
                try:
                    if torch.is_tensor(data) and data.device.type == 'cuda':
                        tensor_size = data.numel()*data.element_size()
                        _end_tensor_offset += tensor_size
                        header[key] = {
                            "dtype": str(data.dtype),                       # JSON cannot stringify torch.Size() type
                            "shape": list(data.shape),
                            "data_offsets": [_start_tensor_offset, _end_tensor_offset],
                            "dbg_tensor_sum": float(torch.sum(data))        # Used for debugging only.
                        }
                        data = data.contiguous()
                        async_copies[key] = {
                            "tensor": data,
                            "tensor_ptr": ctypes.cast(data.data_ptr(), ctypes.POINTER(ctypes.c_byte)),
                            "file_offset": _start_tensor_offset,
                            "tensor_size": tensor_size
                        }
                        _start_tensor_offset = _end_tensor_offset
                        snapshot = f"TENSOR.{key}"
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
                    raise Exception(f"[DataStates][ERROR] Cannot parse {key}, exception: {exc}, data is {data}")

            lean_state_dict = _parse_state("", state_dict)
            lean_state_dict = pickle.dumps(lean_state_dict, protocol=pickle.HIGHEST_PROTOCOL)
            _end_tensor_offset += len(lean_state_dict)
            header.update({"datastates_metadata": {"data_offsets": [_start_tensor_offset, _end_tensor_offset]}})
            header = json.dumps(header).encode("utf-8")
            header_size = len(header).to_bytes(SIZE_UINT64, 'little')   # Force the header size to take 8 bytes
            metadata_size = len(header_size) + len(header)
            with open(path, 'wb') as f:
                f.seek(0)
                f.write(header_size)
                f.write(header)
                # Write the lean state dict towards the end of the file.
                f.seek(_start_tensor_offset+metadata_size)
                f.write(lean_state_dict)
            
            # Launch Async copies
            for _, v in async_copies.items():
                v["file_offset"] += metadata_size
                self.ckpt_engine.ckpt_tensor(version, v["tensor"], v["tensor_size"], v["file_offset"], path)
                        
            return None
        except Exception as exc:
            self.logger.error(f"[DataStates][ERROR] From DataStates save_background, generated exception: {exc}")
            sys.exit(-1)


    def save(self, state_dict, path: str):
        try:
            # self.logger.info(f"Saving {path}")
            self.executor.submit(self.save_background, state_dict, path)
            return True
        except Exception as exc:
            self.logger.info(f"[DataStates][ERROR] Could not save {path}, exception: {exc}, data: {state_dict}")
            sys.exit(-1)
            
        

    def load(self, path: str, map_location=None):
        # self.logger.info(f"[DataStates] Loading checkpoint from {path}...")
        # partition = torch.load(path, map_location=map_location)
        try:
            f = open(path, 'rb')
            f.seek(0)
            header_size_bytes = f.read(SIZE_UINT64)
            header_size = int.from_bytes(header_size_bytes, 'little')
            metadata_size = header_size + SIZE_UINT64
            header = json.loads(f.read(header_size))
            [start_offset, end_offset] = np.add(header["datastates_metadata"]["data_offsets"], metadata_size)
            del(header["datastates_metadata"])
            f.seek(start_offset)
            data = pickle.loads(f.read(end_offset-start_offset))
            try:
                for k, v in header.items():
                    split_k = deque(k.split(KEY_SEPARATOR))
                    dtype = v["dtype"]
                    if dtype.startswith("torch"):
                        dtype = dtype.replace('torch.', '')
                    shape = v["shape"]
                    dbg_tensor_sum = v["dbg_tensor_sum"]
                    [start_offset, end_offset] = np.add(v["data_offsets"], metadata_size)

                    pre_dest = data
                    dest = data
                    while len(split_k):
                        sub_k = split_k.popleft()
                        if sub_k.isdigit():
                            sub_k = int(sub_k) 
                        pre_dest = dest
                        dest = dest[sub_k]
                    if dest != str("TENSOR."+k):
                        raise Exception(f"The key in header {k} does not match key at location {dest}")

                    f.seek(start_offset)
                    tensor_size = end_offset-start_offset
                    byte_data = f.read(tensor_size)
                    tensor_restored = torch.zeros(size=tuple(shape), dtype=getattr(torch, dtype))
                    ctypes.memmove(tensor_restored.data_ptr(), bytes(byte_data), tensor_size)
                    pre_dest[sub_k] = tensor_restored
            except Exception as exc:
                self.logger.error(f"Got error with tensor loading {dtype}, {shape}, {exc}")
                raise Exception(f"Got error with tensor loading {dtype}, {shape}, {exc}")
            self.logger.info(f"[DataStates] Loaded checkpoint from {path}.")
            return data
        except Exception as exc:
            self.logger.info(f"[DataStates][ERROR] Could not load {path}, exception: {exc}")
            sys.exit(-1)


    def commit(self, tag):
        self.wait()
        self.logger.info(f"[DataStates] Checkpoint {tag} is ready now!")
        return True

    def wait(self):
        try:
            t = time.time()
            self.ckpt_engine.wait()
            self.logger.info(f"[DataStates] Wait time in checkpointing engine {time.time()-t}")
        except Exception as exc:
            self.logger.info(f"[DataStates][ERROR] From wait, generated exception: {exc}")
            sys.exit(-1)
        return 
    
    def shutdown(self):
        self.executor.shutdown(True)
        return self.ckpt_engine.shutdown()

    def __del__(self):
        self.shutdown()