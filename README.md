# DataStates I/O Engine
An I/O engine, optimized for GPU-accelerated workload checkpointing, with particular focus on DeepSpeed/Megatron

For detailed description about design principles, implementation, and performance evaluation against state-of-the-art checkpointing engines, please refer [our HPDC'24 paper](https://hal.science/hal-04614247)
> Avinash Maurya, Robert Underwood, M. Mustafa Rafique, Franck Cappello, and Bogdan Nicolae. "DataStates-LLM: Lazy Asynchronous Checkpointing for Large Language Models". HPDC'24: The 33rd International Symposium on High-Performance Parallel and Distributed Computing (Pisa, Italy, 2024).

### Install and test
```
git clone https://github.com/DataStates/datastates-llm.git
cd datastates-llm/
./install.sh

# Test with a simple PyTorch code, DeepSpeed not required.
python tests/python/test_ckpt_engine.py   

# Test with a simple DeepSpeed code.
python tests/python/test_datastates_llm.py   
```
### DataStates Core Engine

The *DataStates Core Engine* is implemented in C++ and provides low-level primitives for checkpointing and restoration, including:

- `ckpt`: Save a memory region to persistent storage.
- `restore`: Load a previously checkpointed region.
- `wait`: Synchronize outstanding checkpoint operations.

These functionalities can be used independently of Python by disabling the Python bindings during CMake configuration.
```bash
cmake -B build -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" -DBUILD_PYTHON_BINDINGS=OFF
cmake --build build -j$(nproc)
cmake --install build
# Test using a simple code.
$INSTALL_PATH/test/test_core_engine
```


### Linking with DeepSpeed
To integrate our asynchronous checkpointing engine with DeepSpeed, a few lines need to be changed in the DeepSpeed repository. While we plan to integrate native support for DataStates-LLM in the official DeepSpeed repository, please use our fork of DeepSpeed at https://github.com/DataStates/DeepSpeed/tree/dev. 