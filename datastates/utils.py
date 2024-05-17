import logging
import sys
import re

HOST_CACHE_SIZE="host_cache_size"
HOST_CACHE_SIZE_DEFAULT=0
CKPT_PARSER_THREADS="parser_threads"
CKPT_PARSER_THREADS_DEFAULT=4
IS_DEEPSPEED_ENABLED="is_deepspeed_enabled"
IS_DEEPSPEED_ENABLED_DEFAULT=False
FAST_CACHE_INIT="fast_cache_init"
FAST_CACHE_INIT_DEFAULT=False
PIN_HOST_CACHE="pin_host_cache"
PIN_HOST_CACHE_DEFAULT=True
SUPPORTED_CONFIG_CLASSES = tuple(["dict", "DeepSpeedConfig"])

def get_logger(logger_name) -> logging.Logger:
    logging_level = logging.INFO
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s")
    channel = logging.StreamHandler(stream=sys.stdout)
    channel.setLevel(logging_level)
    channel.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)
    logger.addHandler(channel)
    return logger

def parse_config(config) -> dict:
    config_class = str(type(config))
    # str(type(config)) can be <class '__main__.DeepSpeedConfig'>, so we just need to extract last word from this string.
    config_class = config_class.split('.')[-1].strip("'>")


    assert config_class in SUPPORTED_CONFIG_CLASSES, f"Config class ({config_class}) not supported. Please use from {SUPPORTED_CONFIG_CLASSES}."
    result = {
        IS_DEEPSPEED_ENABLED: IS_DEEPSPEED_ENABLED_DEFAULT,
        HOST_CACHE_SIZE: HOST_CACHE_SIZE_DEFAULT,
        CKPT_PARSER_THREADS: CKPT_PARSER_THREADS_DEFAULT,
        # In the future, we can give option to do async 
        # memset and allow unpinned host memory
        # FAST_CACHE_INIT: FAST_CACHE_INIT_DEFAULT,
        # PIN_HOST_CACHE: PIN_HOST_CACHE_DEFAULT
    }
    checkpointing_config = {}
    if config_class == "DeepSpeedConfig":
        assert config is not None, "DeepSpeed config cannot be set to None"
        assert config.datastates_config is not None, "DeepSpeed config does not have a datastates_config object"
        assert config.datastates_config.enabled==True, "Datastates is not enabled in DeepSpeed config file"
        
        # Add additional checks about deepspeed, such as which ZeRO stage is supported, 
        # NVMe or CPU offloading of params, optimizer, universal checkpoint loading, etc.
        if hasattr(config, "zero_config") and hasattr(config.zero_config, "offload_param") and config.zero_config.offload_param is not None:
            assert config.zero_config.offload_param.device in (None, "none"), "CPU or NVMe offloaded parameter checkpointing is not yet supported/tested"
        if hasattr(config, "zero_config") and hasattr(config.zero_config, "offload_optimizer") and config.zero_config.offload_optimizer is not None:
            assert config.zero_config.offload_optimizer.device in (None, "none"), "CPU or NVMe offloaded optimizer checkpointing is not yet supported/tested"
        if hasattr(config, "load_universal_checkpoint") and config.load_universal_checkpoint is not None:
            assert config.load_universal_checkpoint is False, "Universal checkpointing loading is not yet supported/tested"
                
        checkpointing_config = config.datastates_config.config 
        result[IS_DEEPSPEED_ENABLED] = True
    elif config_class == "dict":
        checkpointing_config = config         

    for k, _ in result.items():
        if k in checkpointing_config:
            result[k] = checkpointing_config[k]
    return result
    
def get_checkpoint_version(ckpt_path, is_deepspeed_enabled, last_version=-1) -> int:
    version = last_version+1
    deepspeed_folder_pattern = r"/global_step\d+/"
    match = re.search(deepspeed_folder_pattern, ckpt_path)
    if is_deepspeed_enabled and bool(match):
        try:
            # Extract from pathname if using deepspeed, which produces checkpoint in
            # directory path of format `/path-to-ckpt-folder/global_step{version}/filename.pt`
            version = int(ckpt_path.split("/")[-2].replace('global_step', ''))
        except Exception as exc:
            # If not running with deepspeed use a static version number.
            version = last_version+1
    return version

