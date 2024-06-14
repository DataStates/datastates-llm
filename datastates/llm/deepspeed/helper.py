import re

def parse_ds_config(config):
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
    
    return config.datastates_config.config 

def get_ds_checkpoint_version(ckpt_path, last_version=-1) -> int:
    version = last_version+1
    deepspeed_folder_pattern = r"/global_step\d+/"
    match = re.search(deepspeed_folder_pattern, ckpt_path)
    if bool(match):
        try:
            # Extract from pathname if using deepspeed, which produces checkpoint in
            # directory path of format `/path-to-ckpt-folder/global_step{version}/filename.pt`
            version = int(ckpt_path.split("/")[-2].replace('global_step', ''))
        except Exception as exc:
            # If not running with deepspeed use a static version number.
            pass
    return version