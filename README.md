# DataStates-LLM
Optimized checkpointing engine for DeepSpeed/Megatron

For detailed description about design principles, implementation, and performance evaluation against state-of-the-art checkpointing engines, please refer [our HPDC'24 paper](https://hal.science/hal-04614247)
> Avinash Maurya, Robert Underwood, M. Mustafa Rafique, Franck Cappello, and Bogdan Nicolae. "DataStates-LLM: Lazy Asynchronous Checkpointing for Large Language Models". HPDC'24: The 33rd International Symposium on High-Performance Parallel and Distributed Computing (Pisa, Italy, 2024).

### Install and test
```
git clone https://github.com/DataStates/datastates-llm.git
cd datastates-llm/
pip install . -v            # Installs the CPP/Python binding.

# Test with a simple PyTorch code, DeepSpeed not required.
python datastates/tests/test_ckpt_engine.py   

# Test with a simple DeepSpeed code.
python datastates/tests/test_datastates_llm.py   
```

### Linking with DeepSpeed
To integrate our asynchronous checkpointing engine with DeepSpeed, a few lines need to be changed in the DeepSpeed repository. While we plan to integrate native support for DataStates-LLM in the official DeepSpeed repository, please use our fork of DeepSpeed at https://github.com/DataStates/DeepSpeed/tree/dev. 