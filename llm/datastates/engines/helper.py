import ctypes
import sys
from datastates.deepspeed.helper import parse_ds_config, get_ds_checkpoint_version

HOST_CACHE_SIZE="host_cache_size"
HOST_CACHE_SIZE_DEFAULT=1
ENGINE_TYPE="engine_type"
ENGINE_TYPE_DEFAULT="state_engine"
PROFILE_ENGINE="profile_engine"
PROFILE_ENGINE_DEFAULT=False
SUPPORTED_CONFIG_CLASSES = tuple(["dict", "OrderedDict", "DeepSpeedConfig"])
SIZE_UINT64 = ctypes.sizeof(ctypes.c_uint64)
KEY_SEPARATOR = "|"
ALIGNMENT=4096

# Global variables
IS_DEEPSPEED_ENABLED = False

def get_config_type(config) -> str:
    config_class = str(type(config))
    # str(type(config)) can be <class '__main__.DeepSpeedConfig'>, so we just need to extract last word from this string.
    if 'DeepSpeedConfig' in config_class:
        return 'DeepSpeedConfig'
    elif 'dict' in config_class or 'OrderedDict' in config_class:
        return 'dict'
    raise Exception(f"Config class ({config_class}) not supported. Please use from {SUPPORTED_CONFIG_CLASSES}.")

def parse_config(config) -> dict:
    global IS_DEEPSPEED_ENABLED
    config_class = get_config_type(config)
    result = {
        HOST_CACHE_SIZE: HOST_CACHE_SIZE_DEFAULT,
        ENGINE_TYPE: ENGINE_TYPE_DEFAULT,
        PROFILE_ENGINE: PROFILE_ENGINE_DEFAULT,
    }
    
    if config_class == "DeepSpeedConfig":
        checkpointing_config = parse_ds_config(config)
        IS_DEEPSPEED_ENABLED = True
    elif config_class == "dict":
        checkpointing_config = config         

    for k, _ in result.items():
        if k in checkpointing_config:
            result[k] = checkpointing_config[k]
    return result
    
def get_checkpoint_version(ckpt_path, last_version=-1) -> int:
    version = last_version + 1
    if IS_DEEPSPEED_ENABLED:
        version = get_ds_checkpoint_version(ckpt_path, last_version)
    return version

