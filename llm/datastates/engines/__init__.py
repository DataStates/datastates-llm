from .engine_simple import SimpleCheckpointEngine
from .engine_state import StateCheckpointEngine 
from .engine_state_aggregated import StateCheckpointEngineAggregated
from .helper import parse_config
import sys


ENGINE_REGISTRY = {
    "simple_engine": SimpleCheckpointEngine,
    "state_engine": StateCheckpointEngine,
    "state_aggregated_engine": StateCheckpointEngineAggregated,
}

class CheckpointEngine:
    def __init__(self, runtime_config={}, rank=0) -> None:
        try:
            datastates_config = parse_config(runtime_config)

            # Select engine type from datastates_config
            engine_type = datastates_config.get("engine_type", "state_engine")
            if engine_type not in ENGINE_REGISTRY:
                raise ValueError(f"[DataStates.llm] Unknown engine_type '{engine_type}'. "
                                 f"Available: {list(ENGINE_REGISTRY.keys())}")

            engine_cls = ENGINE_REGISTRY[engine_type]
            self._engine = engine_cls(runtime_config, rank)
            print(f"[DataStates.llm] Initialized datastates engine of type {engine_type} on rank {rank} successfully.")

        except Exception as exc:
            print(f"[DataStates.llm][ERROR] Failed to initialize CheckpointEngine: {exc}")
            sys.exit(-1)

    def __getattr__(self, name):
        # Forward all method calls to the underlying engine
        return getattr(self._engine, name)
